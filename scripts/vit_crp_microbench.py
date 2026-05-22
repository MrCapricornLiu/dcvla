#!/usr/bin/env python3
"""Microbenchmark ViT-side reuse vs reuse+prune with fixed inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "transformers" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pytorch-image-models"))
sys.path.insert(0, str(REPO_ROOT / "src" / "openvla"))

from experiments.robot.openvla_utils import get_processor, get_vla  # noqa: E402


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class LoadConfig:
    def __init__(self, checkpoint: str) -> None:
        self.pretrained_checkpoint = checkpoint
        self.load_in_8bit = False
        self.load_in_4bit = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "src" / "openvla" / "checkpoints" / "openvla-7b-finetuned-libero-spatial"),
    )
    parser.add_argument("--task-label", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--skip-count", type=int, default=80)
    parser.add_argument(
        "--skip-counts",
        type=int,
        nargs="+",
        default=None,
        help="Optional grid of skip counts. Overrides --skip-count when set.",
    )
    parser.add_argument("--prune-counts", type=int, nargs="+", default=[0, 8, 24, 48])
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
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


def build_prompt(task_label: str) -> str:
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def clone_cache_item(item: Any) -> Any:
    if item is None:
        return None
    cloned = object.__new__(type(item))
    for name, value in vars(item).items():
        setattr(cloned, name, value.clone() if torch.is_tensor(value) else value)
    return cloned


def clone_cache_state(state: Optional[List[Any]]) -> Optional[List[Any]]:
    if state is None:
        return None
    return [clone_cache_item(item) for item in state]


def make_masks(featurizer: Any, skip_count: int, prune_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_prefix = int(getattr(featurizer, "num_prefix_tokens", 0))
    num_patches = int(featurizer.patch_embed.num_patches)
    seq_len = num_prefix + num_patches
    skip_count = max(0, min(skip_count, num_patches))
    prune_count = max(0, min(prune_count, skip_count))

    reuse_mask = torch.zeros(seq_len, dtype=torch.bool, device=DEVICE)
    delete_mask = torch.zeros(seq_len, dtype=torch.bool, device=DEVICE)
    patch_idx = torch.arange(skip_count, device=DEVICE)
    reuse_mask[num_prefix + patch_idx] = True
    if prune_count > 0:
        delete_mask[num_prefix + patch_idx[:prune_count]] = True
    return reuse_mask, delete_mask


def configure_featurizer(
    featurizer: Any,
    cache_state: Optional[List[Any]],
    reuse_mask: torch.Tensor,
    delete_mask: torch.Tensor,
    enable_reuse: bool,
    enable_delete: bool,
    overhead_benchmark: bool,
) -> None:
    featurizer.set_vla_cache_state(
        cache_state,
        reuse_mask,
        enable_reuse=enable_reuse,
        enable_static_reuse=enable_reuse,
        keyframe_interval=0,
        delete_mask=delete_mask,
        enable_delete=enable_delete,
        overhead_benchmark=overhead_benchmark,
    )


def configure_all_featurizers(
    vla: Any,
    cache_state: Dict[str, Optional[List[Any]]],
    skip_count: int,
    prune_count: int,
    enable_reuse: bool,
    enable_delete: bool,
    overhead_benchmark: bool,
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for name, featurizer in [("alpha", vla.vision_backbone.featurizer)]:
        reuse_mask, delete_mask = make_masks(featurizer, skip_count, prune_count)
        configure_featurizer(
            featurizer,
            clone_cache_state(cache_state.get(name)),
            reuse_mask,
            delete_mask,
            enable_reuse=enable_reuse,
            enable_delete=enable_delete,
            overhead_benchmark=overhead_benchmark,
        )
        stats[name] = {
            "seq_len": int(reuse_mask.numel()),
            "reuse": int(reuse_mask.sum().item()),
            "prune": int(delete_mask.sum().item()),
        }
    if getattr(vla.vision_backbone, "use_fused_vision_backbone", False):
        name = "beta"
        featurizer = vla.vision_backbone.fused_featurizer
        reuse_mask, delete_mask = make_masks(featurizer, skip_count, prune_count)
        configure_featurizer(
            featurizer,
            clone_cache_state(cache_state.get(name)),
            reuse_mask,
            delete_mask,
            enable_reuse=enable_reuse,
            enable_delete=enable_delete,
            overhead_benchmark=overhead_benchmark,
        )
        stats[name] = {
            "seq_len": int(reuse_mask.numel()),
            "reuse": int(reuse_mask.sum().item()),
            "prune": int(delete_mask.sum().item()),
        }
    return stats


@torch.inference_mode()
def run_vision_projector(vla: Any, pixel_values: torch.Tensor) -> torch.Tensor:
    return vla.projector(vla.vision_backbone(pixel_values))


def timed_call(fn) -> Dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None
    wall_start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = float(start_event.elapsed_time(end_event))
    else:
        cuda_ms = 0.0
    return {
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        "cuda_ms": cuda_ms,
        "output_tokens": int(result.shape[1]),
        "output_dim": int(result.shape[2]),
    }


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted({(row["skip_count"], row["variant"], row["prune_count"]) for row in rows})
    summary = []
    for skip_count, variant, prune_count in keys:
        group = [
            row
            for row in rows
            if row["skip_count"] == skip_count
            and row["variant"] == variant
            and row["prune_count"] == prune_count
        ]
        summary.append(
            {
                "skip_count": skip_count,
                "variant": variant,
                "prune_count": prune_count,
                "n": len(group),
                "wall_ms_mean": mean(float(row["wall_ms"]) for row in group),
                "wall_ms_median": median(float(row["wall_ms"]) for row in group),
                "cuda_ms_mean": mean(float(row["cuda_ms"]) for row in group),
                "cuda_ms_median": median(float(row["cuda_ms"]) for row in group),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = str(Path(args.checkpoint).resolve())
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint}")
    old_cwd = Path.cwd()
    try:
        # Match get_vla_action's relative checkpoint assumptions.
        import os

        os.chdir(REPO_ROOT / "src" / "openvla")
        vla = get_vla(LoadConfig(checkpoint)).eval()
        processor = get_processor(LoadConfig(checkpoint))
    finally:
        old_cwd.chdir() if hasattr(old_cwd, "chdir") else None
        import os

        os.chdir(old_cwd)

    image = make_synthetic_image(args.seed)
    inputs = processor(build_prompt(args.task_label), image).to(DEVICE, dtype=torch.bfloat16)
    pixel_values = inputs["pixel_values"]

    zero_cache_state = {"alpha": None, "beta": None}
    configure_all_featurizers(
        vla,
        zero_cache_state,
        skip_count=0,
        prune_count=0,
        enable_reuse=True,
        enable_delete=False,
        overhead_benchmark=False,
    )
    # Fill dense caches once.
    _ = timed_call(lambda: run_vision_projector(vla, pixel_values))
    full_cache_state = {
        "alpha": clone_cache_state(vla.vision_backbone.featurizer.get_vla_cache_state()),
        "beta": clone_cache_state(vla.vision_backbone.fused_featurizer.get_vla_cache_state())
        if getattr(vla.vision_backbone, "use_fused_vision_backbone", False)
        else None,
    }

    rows: List[Dict[str, Any]] = []
    skip_counts = args.skip_counts if args.skip_counts is not None else [args.skip_count]
    variants = []
    seen_variants = set()
    for skip_count in skip_counts:
        for prune_count in args.prune_counts:
            effective_prune = min(prune_count, skip_count)
            key = (skip_count, "reuse_only", effective_prune)
            if key not in seen_variants:
                variants.append((skip_count, "reuse_only", effective_prune, True, False, True))
                seen_variants.add(key)
            if effective_prune > 0:
                key = (skip_count, "reuse_prune", effective_prune)
                if key not in seen_variants:
                    variants.append((skip_count, "reuse_prune", effective_prune, True, True, True))
                    seen_variants.add(key)

    total = (args.warmup + args.iters) * len(variants)
    idx = 0
    mask_stats = None
    for step in range(args.warmup + args.iters):
        for skip_count, variant, prune_count, enable_reuse, enable_delete, overhead in variants:
            idx += 1
            mask_stats = configure_all_featurizers(
                vla,
                full_cache_state,
                skip_count=skip_count,
                prune_count=prune_count,
                enable_reuse=enable_reuse,
                enable_delete=enable_delete,
                overhead_benchmark=overhead,
            )
            row = timed_call(lambda: run_vision_projector(vla, pixel_values))
            row.update({"variant": variant, "skip_count": skip_count, "prune_count": prune_count})
            phase = "warmup" if step < args.warmup else "measure"
            print(
                f"[{idx}/{total}] {phase} skip={skip_count} {variant} prune={prune_count}: "
                f"wall={row['wall_ms']:.3f} cuda={row['cuda_ms']:.3f}"
            )
            if step >= args.warmup:
                rows.append(row)

    summary = summarize(rows)
    print("\nSummary (ms)")
    print("skip | variant | prune | n | wall_mean | wall_median | cuda_mean | cuda_median")
    for item in summary:
        print(
            f"{item['skip_count']} | {item['variant']} | {item['prune_count']} | {item['n']} | "
            f"{item['wall_ms_mean']:.3f} | {item['wall_ms_median']:.3f} | "
            f"{item['cuda_ms_mean']:.3f} | {item['cuda_ms_median']:.3f}"
        )

    payload = {
        "args": vars(args),
        "checkpoint": checkpoint,
        "device": str(DEVICE),
        "mask_stats": mask_stats,
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
