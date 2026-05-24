#!/usr/bin/env python3
"""Probe the soundness assumptions behind LLM-side temporal KV reuse.

This script is intentionally lightweight: it avoids LIBERO rollout and uses
synthetic consecutive frames. For each frame pair, it runs dense OpenVLA forward
passes, constructs the CTR reuse mask from frame difference and previous-step
text-to-vision attention, and compares dense cross-frame KV drift and attention
mass for the tokens selected for reuse vs. recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "transformers" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "openvla"))

from transformers import DynamicCache  # noqa: E402

from experiments.robot.openvla_utils import (  # noqa: E402
    OPENVLA_V01_SYSTEM_PROMPT,
    get_processor,
    get_vla,
)
from experiments.robot.vla_cache_utils import (  # noqa: E402
    find_static_patches,
    get_top_attention_patches,
    token_attention_merge,
)


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
VISUAL_TOKEN_OFFSET = 1
NUM_PATCHES = 256
PATCH_SIZE = 14


@dataclass
class LoadConfig:
    pretrained_checkpoint: str
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "src" / "openvla" / "checkpoints" / "openvla-7b-finetuned-libero-spatial"),
    )
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument(
        "--task-label",
        default="pick up the black bowl between the plate and the ramekin and place it on the plate",
    )
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--changed-patches", type=int, default=24)
    parser.add_argument("--static-top-k", type=int, default=160)
    parser.add_argument("--attention-top-k", type=int, default=160)
    parser.add_argument("--metric", choices=("cosine", "gray_diff", "rgb_diff"), default="gray_diff")
    parser.add_argument("--diff-threshold", type=float, default=0.004)
    parser.add_argument("--sim-threshold", type=float, default=0.996)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=32)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def make_base_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, 224, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, 224, dtype=np.float32)[None, :]
    base = np.stack(
        [
            0.25 + 0.55 * x + 0.10 * y,
            0.20 + 0.25 * x + 0.45 * y,
            0.65 - 0.35 * x + 0.15 * y,
        ],
        axis=-1,
    )
    noise = rng.normal(0, 0.01, size=(224, 224, 3)).astype(np.float32)
    return np.clip(base + noise, 0.0, 1.0)


def make_frame_pair(seed: int, changed_patches: int) -> Tuple[Image.Image, Image.Image, List[int]]:
    rng = np.random.default_rng(seed)
    frame0 = make_base_image(seed)
    frame1 = frame0.copy()
    changed = sorted(rng.choice(NUM_PATCHES, size=min(changed_patches, NUM_PATCHES), replace=False).tolist())
    for patch_id in changed:
        row = patch_id // 16
        col = patch_id % 16
        y0, y1 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
        x0, x1 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
        color = rng.uniform(0.05, 0.95, size=(1, 1, 3)).astype(np.float32)
        frame1[y0:y1, x0:x1, :] = 0.35 * frame1[y0:y1, x0:x1, :] + 0.65 * color
    img0 = Image.fromarray((frame0 * 255).astype(np.uint8), mode="RGB")
    img1 = Image.fromarray((frame1 * 255).astype(np.uint8), mode="RGB")
    return img0, img1, changed


def build_prompt(checkpoint: str, task_label: str) -> str:
    base_name = Path(checkpoint).name
    if "openvla-v01" in base_name:
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def resolve_unnorm_key(vla: Any, task_suite_name: str) -> str:
    stats = getattr(vla, "norm_stats", {})
    for candidate in (task_suite_name, f"{task_suite_name}_no_noops"):
        if candidate in stats:
            return candidate
    if len(stats) == 1:
        return next(iter(stats.keys()))
    raise KeyError(f"Could not resolve unnorm_key for {task_suite_name}; available={list(stats.keys())}")


def detach_cache(cache: DynamicCache) -> DynamicCache:
    if hasattr(cache, "key_cache"):
        cache.key_cache = [tensor.detach() for tensor in cache.key_cache]
    if hasattr(cache, "value_cache"):
        cache.value_cache = [tensor.detach() for tensor in cache.value_cache]
    if hasattr(cache, "cache_position") and cache.cache_position is not None:
        cache.cache_position = cache.cache_position.detach()
    return cache


def reset_llm_cache_flags(vla: Any) -> None:
    config = vla.language_model.config
    config.reusable_patches = None
    config.deleted_patches = None
    config.current_deleted_patches = None
    config.proportion_attn_var = None
    config.vla_cache_effective = False
    config.vla_delete_effective = False
    config.vla_overhead_benchmark = False


@torch.inference_mode()
def dense_action_and_cache(vla: Any, inputs: Dict[str, torch.Tensor], unnorm_key: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    reset_llm_cache_flags(vla)
    prompt_cache = DynamicCache()
    action, caches = vla.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        do_sample=False,
        return_dict_in_generate=True,
        output_attentions=True,
        past_key_values=prompt_cache,
    )
    caches["past_key_values"] = detach_cache(caches["past_key_values"])
    return np.asarray(action, dtype=np.float32), caches


def cache_layer_count(cache: DynamicCache) -> int:
    return min(len(cache.key_cache), len(cache.value_cache))


def layer_token_drift(
    prev_cache: DynamicCache,
    cur_cache: DynamicCache,
    positions: Sequence[int],
    max_layers: int,
) -> Dict[str, float]:
    if not positions:
        return {"kv_drift": 0.0, "k_drift": 0.0, "v_drift": 0.0, "n_positions": 0}

    pos = torch.tensor(list(positions), device=prev_cache.key_cache[0].device, dtype=torch.long)
    k_values: List[torch.Tensor] = []
    v_values: List[torch.Tensor] = []
    for layer_idx in range(min(cache_layer_count(prev_cache), cache_layer_count(cur_cache), max_layers)):
        k0 = prev_cache.key_cache[layer_idx]
        k1 = cur_cache.key_cache[layer_idx]
        v0 = prev_cache.value_cache[layer_idx]
        v1 = cur_cache.value_cache[layer_idx]
        max_len = min(k0.shape[2], k1.shape[2], v0.shape[2], v1.shape[2])
        valid = pos[pos < max_len]
        if valid.numel() == 0:
            continue
        k_delta = (k1.index_select(2, valid) - k0.index_select(2, valid)).float()
        v_delta = (v1.index_select(2, valid) - v0.index_select(2, valid)).float()
        k_values.append(k_delta.pow(2).mean(dim=(0, 1, 3)).sqrt().detach().cpu())
        v_values.append(v_delta.pow(2).mean(dim=(0, 1, 3)).sqrt().detach().cpu())

    if not k_values:
        return {"kv_drift": 0.0, "k_drift": 0.0, "v_drift": 0.0, "n_positions": len(positions)}

    k = torch.cat(k_values)
    v = torch.cat(v_values)
    return {
        "kv_drift": float(((k + v) * 0.5).mean().item()),
        "k_drift": float(k.mean().item()),
        "v_drift": float(v.mean().item()),
        "n_positions": len(positions),
    }


def attention_stats(attn_scores: torch.Tensor, patch_ids: Sequence[int]) -> Dict[str, float]:
    if not patch_ids:
        return {"attn_mean": 0.0, "attn_sum": 0.0, "n_positions": 0}
    scores = attn_scores.detach().float().cpu().reshape(-1)
    ids = torch.tensor([pid for pid in patch_ids if 0 <= pid < scores.numel()], dtype=torch.long)
    if ids.numel() == 0:
        return {"attn_mean": 0.0, "attn_sum": 0.0, "n_positions": len(patch_ids)}
    selected = scores.index_select(0, ids)
    return {
        "attn_mean": float(selected.mean().item()),
        "attn_sum": float(selected.sum().item()),
        "n_positions": len(patch_ids),
    }


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLA_LLM_DETAIL_PROFILE", "0")
    os.environ.setdefault("VLA_VIT_DETAIL_PROFILE", "0")
    os.environ.setdefault("VLA_LLM_PROFILE", "0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = str(Path(args.checkpoint).resolve())
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint}")
    cfg = LoadConfig(pretrained_checkpoint=checkpoint)

    old_cwd = Path.cwd()
    os.chdir(REPO_ROOT / "src" / "openvla")
    try:
        vla = get_vla(cfg).eval()
        processor = get_processor(cfg)
    finally:
        os.chdir(old_cwd)
    reset_llm_cache_flags(vla)

    unnorm_key = resolve_unnorm_key(vla, args.task_suite_name)
    prompt = build_prompt(checkpoint, args.task_label)
    rows: List[Dict[str, Any]] = []

    for pair_idx in range(args.pairs):
        img0, img1, changed = make_frame_pair(args.seed + pair_idx, args.changed_patches)
        inputs0 = processor(prompt, img0).to(DEVICE, dtype=torch.bfloat16)
        inputs1 = processor(prompt, img1).to(DEVICE, dtype=torch.bfloat16)

        print(f"[{pair_idx + 1}/{args.pairs}] dense frame t-1")
        action0, cache0 = dense_action_and_cache(vla, inputs0, unnorm_key)
        print(f"[{pair_idx + 1}/{args.pairs}] dense frame t")
        action1, cache1 = dense_action_and_cache(vla, inputs1, unnorm_key)

        attn_scores = token_attention_merge(cache0["attentions"])
        top_attn = set(get_top_attention_patches(attn_scores, args.attention_top_k))
        static = set(
            find_static_patches(
                img0,
                img1,
                top_k=args.static_top_k,
                metric=args.metric,
                sim_threshold=args.sim_threshold,
                diff_threshold=args.diff_threshold,
            )
        )
        all_ids = set(range(NUM_PATCHES))
        recompute_ids = sorted((all_ids - static) | top_attn)
        reuse_ids = sorted(all_ids - set(recompute_ids))
        reuse_positions = [pid + VISUAL_TOKEN_OFFSET for pid in reuse_ids]
        recompute_positions = [pid + VISUAL_TOKEN_OFFSET for pid in recompute_ids]

        reuse_drift = layer_token_drift(cache0["past_key_values"], cache1["past_key_values"], reuse_positions, args.max_layers)
        recompute_drift = layer_token_drift(cache0["past_key_values"], cache1["past_key_values"], recompute_positions, args.max_layers)
        reuse_attn = attention_stats(attn_scores, reuse_ids)
        recompute_attn = attention_stats(attn_scores, recompute_ids)
        action_l2 = float(np.linalg.norm(action1 - action0))

        row = {
            "pair": pair_idx,
            "changed_patch_count": len(changed),
            "static_candidate_count": len(static),
            "attention_top_k": args.attention_top_k,
            "reuse_count": len(reuse_ids),
            "recompute_count": len(recompute_ids),
            "reuse_kv_drift": reuse_drift["kv_drift"],
            "recompute_kv_drift": recompute_drift["kv_drift"],
            "reuse_k_drift": reuse_drift["k_drift"],
            "recompute_k_drift": recompute_drift["k_drift"],
            "reuse_v_drift": reuse_drift["v_drift"],
            "recompute_v_drift": recompute_drift["v_drift"],
            "reuse_attn_mean": reuse_attn["attn_mean"],
            "recompute_attn_mean": recompute_attn["attn_mean"],
            "reuse_attn_sum": reuse_attn["attn_sum"],
            "recompute_attn_sum": recompute_attn["attn_sum"],
            "reuse_attn_weighted_kv_drift": reuse_drift["kv_drift"] * reuse_attn["attn_mean"],
            "recompute_attn_weighted_kv_drift": recompute_drift["kv_drift"] * recompute_attn["attn_mean"],
            "reuse_vs_recompute_kv_drift_ratio": safe_ratio(reuse_drift["kv_drift"], recompute_drift["kv_drift"]),
            "reuse_vs_recompute_attn_mean_ratio": safe_ratio(reuse_attn["attn_mean"], recompute_attn["attn_mean"]),
            "dense_action_l2_between_frames": action_l2,
        }
        rows.append(row)
        print(
            "  reuse/recompute="
            f"{row['reuse_count']}/{row['recompute_count']} "
            f"kv_drift={row['reuse_kv_drift']:.4f}/{row['recompute_kv_drift']:.4f} "
            f"attn_mean={row['reuse_attn_mean']:.6f}/{row['recompute_attn_mean']:.6f}"
        )

    summary: Dict[str, Any] = {
        "n_pairs": len(rows),
        "reuse_count_mean": mean(row["reuse_count"] for row in rows),
        "recompute_count_mean": mean(row["recompute_count"] for row in rows),
        "reuse_kv_drift_mean": mean(row["reuse_kv_drift"] for row in rows),
        "recompute_kv_drift_mean": mean(row["recompute_kv_drift"] for row in rows),
        "reuse_k_drift_mean": mean(row["reuse_k_drift"] for row in rows),
        "recompute_k_drift_mean": mean(row["recompute_k_drift"] for row in rows),
        "reuse_v_drift_mean": mean(row["reuse_v_drift"] for row in rows),
        "recompute_v_drift_mean": mean(row["recompute_v_drift"] for row in rows),
        "reuse_attn_mean": mean(row["reuse_attn_mean"] for row in rows),
        "recompute_attn_mean": mean(row["recompute_attn_mean"] for row in rows),
        "reuse_attn_sum_mean": mean(row["reuse_attn_sum"] for row in rows),
        "recompute_attn_sum_mean": mean(row["recompute_attn_sum"] for row in rows),
        "reuse_attn_weighted_kv_drift_mean": mean(row["reuse_attn_weighted_kv_drift"] for row in rows),
        "recompute_attn_weighted_kv_drift_mean": mean(row["recompute_attn_weighted_kv_drift"] for row in rows),
        "dense_action_l2_between_frames_mean": mean(row["dense_action_l2_between_frames"] for row in rows),
    }
    summary["reuse_vs_recompute_kv_drift_ratio"] = safe_ratio(
        summary["reuse_kv_drift_mean"], summary["recompute_kv_drift_mean"]
    )
    summary["reuse_vs_recompute_attn_mean_ratio"] = safe_ratio(
        summary["reuse_attn_mean"], summary["recompute_attn_mean"]
    )
    summary["reuse_vs_recompute_attn_weighted_kv_drift_ratio"] = safe_ratio(
        summary["reuse_attn_weighted_kv_drift_mean"],
        summary["recompute_attn_weighted_kv_drift_mean"],
    )

    payload = {
        "args": vars(args),
        "checkpoint": checkpoint,
        "device": str(DEVICE),
        "unnorm_key": unnorm_key,
        "summary": summary,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSummary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
