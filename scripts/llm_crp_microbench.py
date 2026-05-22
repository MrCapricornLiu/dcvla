#!/usr/bin/env python3
"""Microbenchmark OpenVLA LLM-side CRP with fixed inputs and cache.

This benchmark avoids LIBERO and simulation overhead. It builds one fixed prompt,
one fixed image, one warmed prompt cache, and one fixed CRP mask, then compares
overhead-controlled baseline execution against effective CRP execution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

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
    get_layer_mask_schedule,
    task_relevant_selection,
    token_attention_merge,
)


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class LoadConfig:
    pretrained_checkpoint: str
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class Variant:
    name: str
    cache_effective: bool
    prune_effective: bool
    output_attentions: bool
    overhead_benchmark: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "src" / "openvla" / "checkpoints" / "openvla-7b-finetuned-libero-spatial"),
        help="OpenVLA checkpoint path.",
    )
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument(
        "--task-label",
        default="pick up the black bowl between the plate and the ramekin and place it on the plate",
    )
    parser.add_argument("--image", default=None, help="Optional RGB image path. Uses a synthetic image by default.")
    parser.add_argument("--mode", choices=("prefill", "generate", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--static-top-k", type=int, default=130)
    parser.add_argument("--attention-top-k", type=int, default=120)
    parser.add_argument("--prune-ratio", type=float, default=0.10)
    parser.add_argument(
        "--target-reuse-count",
        type=int,
        default=None,
        help="Override mask construction to use this many reuse positions, approximating rollout ratios.",
    )
    parser.add_argument(
        "--target-prune-count",
        type=int,
        default=None,
        help="Override mask construction to use this many prune positions, approximating rollout ratios.",
    )
    parser.add_argument(
        "--target-skip-count",
        type=int,
        default=None,
        help="Override total reuse+prune count. If set with --target-prune-count, reuse is inferred.",
    )
    parser.add_argument("--patch-metric", choices=("cosine", "gray_diff", "rgb_diff"), default="cosine")
    parser.add_argument("--sim-threshold", type=float, default=0.996)
    parser.add_argument("--diff-threshold", type=float, default=0.10)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--empty-cache-between-runs",
        action="store_true",
        help="Call torch.cuda.empty_cache() before each timed run. Off by default to match continuous inference.",
    )
    return parser.parse_args()


def make_synthetic_image(seed: int) -> Image.Image:
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
    noise = rng.normal(0, 0.015, size=(224, 224, 3)).astype(np.float32)
    image = np.clip(base + noise, 0.0, 1.0)
    return Image.fromarray((image * 255).astype(np.uint8), mode="RGB")


def load_image(path: Optional[str], seed: int) -> Image.Image:
    if path is None:
        return make_synthetic_image(seed)
    return Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)


def build_prompt(checkpoint: str, task_label: str) -> str:
    base_name = Path(checkpoint).name
    if "openvla-v01" in base_name:
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def append_action_prompt_token(input_ids: torch.Tensor) -> torch.Tensor:
    if torch.all(input_ids[:, -1] == 29871):
        return input_ids
    suffix = torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)
    return torch.cat((input_ids, suffix), dim=1)


def resolve_unnorm_key(vla: Any, task_suite_name: str) -> str:
    stats = getattr(vla, "norm_stats", {})
    candidates = [task_suite_name, f"{task_suite_name}_no_noops"]
    for candidate in candidates:
        if candidate in stats:
            return candidate
    if len(stats) == 1:
        return next(iter(stats.keys()))
    raise KeyError(f"Could not resolve unnorm_key from candidates {candidates}; available={list(stats.keys())}")


def clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    cloned = DynamicCache()
    if hasattr(cache, "_seen_tokens"):
        cloned._seen_tokens = cache._seen_tokens
    if hasattr(cache, "key_cache"):
        cloned.key_cache = [tensor.clone() for tensor in cache.key_cache]
    if hasattr(cache, "value_cache"):
        cloned.value_cache = [tensor.clone() for tensor in cache.value_cache]
    if hasattr(cache, "cache_position"):
        cache_position = cache.cache_position
        cloned.cache_position = None if cache_position is None else cache_position.clone()
    return cloned


def detach_cache(cache: DynamicCache) -> DynamicCache:
    if hasattr(cache, "key_cache"):
        cache.key_cache = [tensor.detach() for tensor in cache.key_cache]
    if hasattr(cache, "value_cache"):
        cache.value_cache = [tensor.detach() for tensor in cache.value_cache]
    if hasattr(cache, "cache_position") and cache.cache_position is not None:
        cache.cache_position = cache.cache_position.detach()
    return cache


def tensor_numel(value: Optional[torch.Tensor]) -> int:
    if value is None:
        return 0
    return int(value.numel())


def configure_llm_crp(vla: Any, variant: Variant, reusable: torch.Tensor, pruned: torch.Tensor, schedule: torch.Tensor) -> None:
    config = vla.language_model.config
    config.reusable_patches = reusable
    config.deleted_patches = pruned
    config.current_deleted_patches = None
    config.proportion_attn_var = schedule
    config.vla_cache_effective = bool(variant.cache_effective)
    config.vla_delete_effective = bool(variant.prune_effective)
    config.vla_overhead_benchmark = bool(variant.overhead_benchmark)


def reset_llm_crp(vla: Any) -> None:
    config = vla.language_model.config
    config.reusable_patches = None
    config.deleted_patches = None
    config.current_deleted_patches = None
    config.proportion_attn_var = None
    config.vla_cache_effective = False
    config.vla_delete_effective = False
    config.vla_overhead_benchmark = False


def elapsed_ms(start_event: Optional[torch.cuda.Event], wall_start: float) -> Dict[str, float]:
    if torch.cuda.is_available():
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = float(start_event.elapsed_time(end_event)) if start_event is not None else 0.0
    else:
        cuda_ms = 0.0
    return {"outer_wall_ms": (time.perf_counter() - wall_start) * 1000.0, "outer_cuda_ms": cuda_ms}


def timer_start() -> tuple[float, Optional[torch.cuda.Event]]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = None
    return time.perf_counter(), start_event


def collect_metrics(vla: Any, elapsed: Dict[str, float], mode: str, variant: Variant) -> Dict[str, float | str | bool]:
    metrics = dict(getattr(vla, "_vla_latency_metrics", {}) or {})
    result: Dict[str, float | str | bool] = {
        "mode": mode,
        "variant": variant.name,
        "cache_effective": variant.cache_effective,
        "prune_effective": variant.prune_effective,
        "output_attentions": variant.output_attentions,
        **elapsed,
    }
    for key, value in metrics.items():
        result[key] = float(value)
    return result


@torch.inference_mode()
def warm_reference_cache(vla: Any, inputs: Dict[str, torch.Tensor], unnorm_key: str) -> Dict[str, Any]:
    reset_llm_crp(vla)
    prompt_cache = DynamicCache()
    action, last_caches = vla.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        do_sample=False,
        return_dict_in_generate=True,
        output_attentions=True,
        past_key_values=prompt_cache,
    )
    del action
    last_caches["past_key_values"] = detach_cache(last_caches["past_key_values"])
    return last_caches


def build_masks(
    prev_attn: Iterable[Any],
    image: Image.Image,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    stable_patches = find_static_patches(
        image,
        image,
        top_k=args.static_top_k,
        metric=args.patch_metric,
        sim_threshold=args.sim_threshold,
        diff_threshold=args.diff_threshold,
    )
    _, reusable_positions, pruned_positions = task_relevant_selection(
        prev_attn,
        image,
        stable_patches,
        top_k=args.attention_top_k,
        delete_ratio=args.prune_ratio,
        return_delete=True,
    )
    target_reuse = args.target_reuse_count
    target_prune = args.target_prune_count
    if args.target_skip_count is not None:
        target_prune = 0 if target_prune is None else target_prune
        target_reuse = max(0, args.target_skip_count - target_prune)

    target_fill_count = 0
    if target_reuse is not None or target_prune is not None:
        target_reuse = max(0, int(target_reuse or 0))
        target_prune = max(0, int(target_prune or 0))
        target_skip = min(256, target_reuse + target_prune)
        attn_score = token_attention_merge(prev_attn)
        attn_values = attn_score.cpu().numpy() if isinstance(attn_score, torch.Tensor) else attn_score
        top_patches = set(get_top_attention_patches(attn_score, args.attention_top_k))
        stable_candidates = sorted(set(stable_patches) - top_patches)
        fallback_candidates = sorted(set(range(256)) - set(stable_candidates) - top_patches)
        stable_candidates.sort(key=lambda pid: float(attn_values[pid]))
        fallback_candidates.sort(key=lambda pid: float(attn_values[pid]))
        selected_patch_ids = stable_candidates[:target_skip]
        if len(selected_patch_ids) < target_skip:
            needed = target_skip - len(selected_patch_ids)
            target_fill_count = needed
            selected_patch_ids.extend(fallback_candidates[:needed])
        selected_patch_ids = selected_patch_ids[:target_skip]
        selected_patch_ids.sort(key=lambda pid: float(attn_values[pid]))
        pruned_patch_ids = selected_patch_ids[: min(target_prune, len(selected_patch_ids))]
        pruned_set = set(pruned_patch_ids)
        reusable_patch_ids = [pid for pid in selected_patch_ids if pid not in pruned_set]
        reusable_positions = sorted(pid + 1 for pid in reusable_patch_ids)
        pruned_positions = sorted(pid + 1 for pid in pruned_patch_ids)

    reusable = torch.tensor(reusable_positions, device=DEVICE, dtype=torch.long) if reusable_positions else torch.empty(0, device=DEVICE, dtype=torch.long)
    pruned = torch.tensor(pruned_positions, device=DEVICE, dtype=torch.long) if pruned_positions else torch.empty(0, device=DEVICE, dtype=torch.long)
    schedule = get_layer_mask_schedule(prev_attn).to(device=DEVICE, dtype=torch.float32)
    return {
        "stable_patch_count": len(stable_patches),
        "target_fill_count": target_fill_count,
        "reusable_positions": reusable,
        "pruned_positions": pruned,
        "schedule": schedule,
    }


@torch.inference_mode()
def run_prefill_once(
    vla: Any,
    inputs: Dict[str, torch.Tensor],
    prompt_cache: DynamicCache,
    variant: Variant,
    reusable: torch.Tensor,
    pruned: torch.Tensor,
    schedule: torch.Tensor,
) -> Dict[str, float | str | bool]:
    configure_llm_crp(vla, variant, reusable, pruned, schedule)
    cloned_cache = clone_dynamic_cache(prompt_cache)
    vla._reset_latency_metrics()
    wall_start, start_event = timer_start()
    outputs = vla(
        input_ids=append_action_prompt_token(inputs["input_ids"]),
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs.get("attention_mask"),
        past_key_values=cloned_cache,
        use_cache=True,
        return_dict=True,
        output_attentions=variant.output_attentions,
    )
    elapsed = elapsed_ms(start_event, wall_start)
    extra = {
        "logits_seq_len": int(outputs.logits.shape[1]) if getattr(outputs, "logits", None) is not None else 0,
    }
    del outputs
    reset_llm_crp(vla)
    row = collect_metrics(vla, elapsed, "prefill", variant)
    row.update(extra)
    return row


@torch.inference_mode()
def run_generate_once(
    vla: Any,
    inputs: Dict[str, torch.Tensor],
    prompt_cache: DynamicCache,
    variant: Variant,
    reusable: torch.Tensor,
    pruned: torch.Tensor,
    schedule: torch.Tensor,
    max_new_tokens: int,
) -> Dict[str, float | str | bool]:
    configure_llm_crp(vla, variant, reusable, pruned, schedule)
    cloned_cache = clone_dynamic_cache(prompt_cache)
    vla._reset_latency_metrics()
    wall_start, start_event = timer_start()
    outputs = vla.generate(
        append_action_prompt_token(inputs["input_ids"]),
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs.get("attention_mask"),
        past_key_values=cloned_cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_attentions=variant.output_attentions,
    )
    elapsed = elapsed_ms(start_event, wall_start)
    extra = {
        "generated_seq_len": int(outputs.sequences.shape[1]) if getattr(outputs, "sequences", None) is not None else 0,
    }
    del outputs
    reset_llm_crp(vla)
    row = collect_metrics(vla, elapsed, "generate", variant)
    row.update(extra)
    return row


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["variant"]), []).append(row)

    summary = []
    metric_keys = [
        "outer_wall_ms",
        "outer_cuda_ms",
        "vit_wall_ms",
        "vit_cuda_ms",
        "llm_wall_ms",
        "llm_cuda_ms",
    ]
    for (mode, variant), group in sorted(grouped.items()):
        item: Dict[str, Any] = {"mode": mode, "variant": variant, "n": len(group)}
        for key in metric_keys:
            values = [float(row.get(key, 0.0)) for row in group]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_median"] = median(values)
        summary.append(item)
    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    headers = [
        "mode",
        "variant",
        "n",
        "outer_wall_mean",
        "outer_cuda_mean",
        "llm_wall_mean",
        "llm_cuda_mean",
        "vit_wall_mean",
    ]
    print("\nSummary (ms)")
    print(" | ".join(headers))
    for item in summary:
        print(
            " | ".join(
                [
                    str(item["mode"]),
                    str(item["variant"]),
                    str(item["n"]),
                    f"{item['outer_wall_ms_mean']:.3f}",
                    f"{item['outer_cuda_ms_mean']:.3f}",
                    f"{item['llm_wall_ms_mean']:.3f}",
                    f"{item['llm_cuda_ms_mean']:.3f}",
                    f"{item['vit_wall_ms_mean']:.3f}",
                ]
            )
        )


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
    unnorm_key = resolve_unnorm_key(vla, args.task_suite_name)
    max_new_tokens = args.max_new_tokens or vla.get_action_dim(unnorm_key)

    image = load_image(args.image, args.seed)
    prompt = build_prompt(checkpoint, args.task_label)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    print(f"Prompt tokens: {inputs['input_ids'].shape[-1]}")
    print(f"Action tokens: {max_new_tokens}")
    print(f"Unnorm key: {unnorm_key}")
    print(f"LLM attention class: {type(vla.language_model.model.layers[0].self_attn).__name__}")

    print("Warming reference cache and attention masks...")
    reference = warm_reference_cache(vla, inputs, unnorm_key)
    prompt_cache = reference["past_key_values"]
    masks = build_masks(reference["attentions"], image, args)
    reusable = masks["reusable_positions"]
    pruned = masks["pruned_positions"]
    schedule = masks["schedule"]
    print(
        "Mask stats: "
        f"stable={masks['stable_patch_count']} reuse={tensor_numel(reusable)} "
        f"prune={tensor_numel(pruned)} schedule_layers={tensor_numel(schedule)} "
        f"schedule_mean={float(schedule.mean().detach().cpu()):.3f}"
    )

    variants = [
        Variant("baseline_attn_true", cache_effective=False, prune_effective=False, output_attentions=True),
        Variant("crp_attn_true", cache_effective=True, prune_effective=True, output_attentions=True),
        Variant("baseline_attn_false", cache_effective=False, prune_effective=False, output_attentions=False),
        Variant("crp_attn_false", cache_effective=True, prune_effective=True, output_attentions=False),
    ]
    modes = ["prefill", "generate"] if args.mode == "both" else [args.mode]
    rows: List[Dict[str, Any]] = []

    total_runs = (args.warmup + args.iters) * len(variants) * len(modes)
    run_idx = 0
    for idx in range(args.warmup + args.iters):
        for mode in modes:
            for variant in variants:
                run_idx += 1
                if args.empty_cache_between_runs and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if mode == "prefill":
                    row = run_prefill_once(vla, inputs, prompt_cache, variant, reusable, pruned, schedule)
                else:
                    row = run_generate_once(
                        vla,
                        inputs,
                        prompt_cache,
                        variant,
                        reusable,
                        pruned,
                        schedule,
                        max_new_tokens,
                    )
                phase = "warmup" if idx < args.warmup else "measure"
                print(
                    f"[{run_idx}/{total_runs}] {phase} {mode} {variant.name}: "
                    f"outer={row['outer_wall_ms']:.3f} ms llm={row.get('llm_wall_ms', 0.0):.3f} ms"
                )
                if idx >= args.warmup:
                    rows.append(row)

    summary = summarize(rows)
    print_summary(summary)

    payload = {
        "args": vars(args),
        "checkpoint": checkpoint,
        "device": str(DEVICE),
        "unnorm_key": unnorm_key,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "max_new_tokens": int(max_new_tokens),
        "attention_class": type(vla.language_model.model.layers[0].self_attn).__name__,
        "mask_stats": {
            "stable_patch_count": masks["stable_patch_count"],
            "target_fill_count": masks["target_fill_count"],
            "reuse_count": tensor_numel(reusable),
            "prune_count": tensor_numel(pruned),
            "schedule_layers": tensor_numel(schedule),
            "schedule_mean": float(schedule.mean().detach().cpu()),
        },
        "rows": rows,
        "summary": summary,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
