#!/usr/bin/env python3
"""Oracle microbenchmarks for ideal compact-token ViT and LLM execution.

These benchmarks do not exercise the CRP implementation. They answer a simpler
question: if the model could run directly on a shorter dense sequence without
masking, gather/scatter, or cache-update overhead, how much CUDA time would the
hardware actually save?
"""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--bench", choices=("vit", "llm", "both"), default="both")
    parser.add_argument("--task-label", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--skip-counts", type=int, nargs="+", default=[0, 75, 128, 192, 224, 250])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument(
        "--llm-attention-modes",
        choices=("false", "true", "both"),
        default="both",
        help="Whether to run the LLM oracle with output_attentions disabled, enabled, or both.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
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


def append_action_prompt_token(input_ids: torch.Tensor) -> torch.Tensor:
    if torch.all(input_ids[:, -1] == 29871):
        return input_ids
    suffix = torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)
    return torch.cat((input_ids, suffix), dim=1)


def timer_start() -> tuple[float, Optional[torch.cuda.Event]]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = None
    return time.perf_counter(), start_event


def timer_stop(wall_start: float, start_event: Optional[torch.cuda.Event]) -> Dict[str, float]:
    if torch.cuda.is_available():
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = float(start_event.elapsed_time(end_event)) if start_event is not None else 0.0
    else:
        cuda_ms = 0.0
    return {"wall_ms": (time.perf_counter() - wall_start) * 1000.0, "cuda_ms": cuda_ms}


def summarize(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    summary = []
    for key, group in sorted(groups.items()):
        item = {name: value for name, value in zip(group_keys, key)}
        item["n"] = len(group)
        for metric in ("wall_ms", "cuda_ms"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_median"] = median(values)
        summary.append(item)
    return summary


def speedup_from_baseline(summary: List[Dict[str, Any]], group_key: str = "branch") -> None:
    baselines: Dict[Any, float] = {}
    for item in summary:
        if item["skip_count"] == 0:
            baselines[item.get(group_key)] = float(item["cuda_ms_mean"])
    for item in summary:
        base = baselines.get(item.get(group_key))
        if base and base > 0:
            current = float(item["cuda_ms_mean"])
            item["cuda_speedup_pct"] = (base - current) / base * 100.0
            item["cuda_speedup_x"] = base / current if current > 0 else 0.0


def reset_featurizer_cache(featurizer: Any) -> None:
    if hasattr(featurizer, "reset_vla_cache"):
        featurizer.reset_vla_cache()
    for block in getattr(featurizer, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "set_cache_context"):
            attn.set_cache_context(None)


def vit_branch_specs(vla: Any) -> List[tuple[str, Any]]:
    specs = [("alpha", vla.vision_backbone.featurizer)]
    if getattr(vla.vision_backbone, "use_fused_vision_backbone", False):
        specs.append(("beta", vla.vision_backbone.fused_featurizer))
    return specs


@torch.inference_mode()
def run_vit_oracle(vla: Any, args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    tensors: Dict[tuple[str, int], torch.Tensor] = {}
    branch_meta: Dict[str, Dict[str, int]] = {}

    for branch, featurizer in vit_branch_specs(vla):
        reset_featurizer_cache(featurizer)
        num_prefix = int(getattr(featurizer, "num_prefix_tokens", 0))
        num_patches = int(featurizer.patch_embed.num_patches)
        embed_dim = int(getattr(featurizer, "embed_dim"))
        branch_meta[branch] = {
            "num_prefix": num_prefix,
            "num_patches": num_patches,
            "full_seq_len": num_prefix + num_patches,
            "embed_dim": embed_dim,
            "num_blocks": len(featurizer.blocks),
        }
        dtype = next(featurizer.parameters()).dtype
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(args.seed + len(branch_meta))
        for skip_count in args.skip_counts:
            effective_skip = max(0, min(int(skip_count), num_patches))
            seq_len = num_prefix + num_patches - effective_skip
            tensors[(branch, skip_count)] = torch.randn(
                1,
                seq_len,
                embed_dim,
                device=DEVICE,
                dtype=dtype,
                generator=generator,
            )

    total = (args.warmup + args.iters) * len(args.skip_counts)
    for step in range(args.warmup + args.iters):
        for run_idx, skip_count in enumerate(args.skip_counts, start=1):
            phase = "warmup" if step < args.warmup else "measure"

            branch_times: Dict[str, Dict[str, float]] = {}
            wall_start, start_event = timer_start()
            for branch, featurizer in vit_branch_specs(vla):
                x = tensors[(branch, skip_count)]
                branch_wall_start, branch_event = timer_start()
                y = featurizer.blocks(x)
                if hasattr(featurizer, "norm"):
                    y = featurizer.norm(y)
                branch_times[branch] = timer_stop(branch_wall_start, branch_event)
                del y
            combined = timer_stop(wall_start, start_event)

            print(
                f"[ViT {step * len(args.skip_counts) + run_idx}/{total}] {phase} "
                f"skip={skip_count}: combined_cuda={combined['cuda_ms']:.3f} ms"
            )
            if step >= args.warmup:
                rows.append(
                    {
                        "branch": "combined",
                        "skip_count": int(skip_count),
                        "seq_len": -1,
                        **combined,
                    }
                )
                for branch, timing in branch_times.items():
                    meta = branch_meta[branch]
                    effective_skip = max(0, min(int(skip_count), meta["num_patches"]))
                    rows.append(
                        {
                            "branch": branch,
                            "skip_count": int(skip_count),
                            "seq_len": meta["full_seq_len"] - effective_skip,
                            **timing,
                        }
                    )

    summary = summarize(rows, ["branch", "skip_count", "seq_len"])
    speedup_from_baseline(summary, group_key="branch")
    return {"branch_meta": branch_meta, "rows": rows, "summary": summary}


@torch.inference_mode()
def infer_multimodal_length(vla: Any, processor: Any, args: argparse.Namespace) -> Dict[str, int]:
    image = make_synthetic_image(args.seed)
    inputs = processor(build_prompt(args.task_label), image).to(DEVICE, dtype=torch.bfloat16)
    input_ids = append_action_prompt_token(inputs["input_ids"])
    reset_llm_crp(vla)
    projected = vla.projector(vla.vision_backbone(inputs["pixel_values"]))
    return {
        "text_tokens": int(input_ids.shape[-1]),
        "vision_tokens": int(projected.shape[1]),
        "full_seq_len": int(input_ids.shape[-1] + projected.shape[1]),
        "hidden_size": int(projected.shape[-1]),
    }


def reset_llm_crp(vla: Any) -> None:
    config = vla.language_model.config
    config.reusable_patches = None
    config.deleted_patches = None
    config.current_deleted_patches = None
    config.current_visible_kv_indices = None
    config.proportion_attn_var = None
    config.vla_cache_effective = False
    config.vla_delete_effective = False
    config.vla_overhead_benchmark = False


def build_4d_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    min_dtype = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device)
    if seq_len != 1:
        mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


@torch.inference_mode()
def run_llm_decoder_dense(
    vla: Any,
    hidden_states: torch.Tensor,
    output_attentions: bool,
) -> torch.Tensor:
    model = vla.language_model.model
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    cache_position = position_ids.squeeze(0)
    causal_mask = (
        build_4d_causal_mask(seq_len, hidden_states.dtype, hidden_states.device)
        if output_attentions
        else None
    )

    for decoder_layer in model.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=output_attentions,
            use_cache=False,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]

    hidden_states = model.norm(hidden_states)
    return vla.language_model.lm_head(hidden_states)


@torch.inference_mode()
def run_llm_oracle(vla: Any, processor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    reset_llm_crp(vla)
    length_meta = infer_multimodal_length(vla, processor, args)
    full_len = length_meta["full_seq_len"]
    hidden_size = int(vla.language_model.config.hidden_size)
    dtype = next(vla.language_model.parameters()).dtype
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(args.seed + 1000)

    tensors: Dict[int, torch.Tensor] = {}
    masks: Dict[int, torch.Tensor] = {}
    length_by_skip: Dict[int, int] = {}
    for skip_count in args.skip_counts:
        effective_skip = max(0, min(int(skip_count), length_meta["vision_tokens"]))
        seq_len = full_len - effective_skip
        length_by_skip[int(skip_count)] = seq_len
        tensors[int(skip_count)] = torch.randn(
            1,
            seq_len,
            hidden_size,
            device=DEVICE,
            dtype=dtype,
            generator=generator,
        )
        masks[int(skip_count)] = torch.ones(1, seq_len, dtype=torch.long, device=DEVICE)

    rows: List[Dict[str, Any]] = []
    attention_modes = [False, True] if args.llm_attention_modes == "both" else [args.llm_attention_modes == "true"]
    total = (args.warmup + args.iters) * len(args.skip_counts) * len(attention_modes)
    for step in range(args.warmup + args.iters):
        for mode_idx, output_attentions in enumerate(attention_modes):
            for run_idx, skip_count in enumerate(args.skip_counts, start=1):
                phase = "warmup" if step < args.warmup else "measure"
                hidden = tensors[int(skip_count)]
                wall_start, start_event = timer_start()
                outputs = run_llm_decoder_dense(vla, hidden, output_attentions)
                timing = timer_stop(wall_start, start_event)
                del outputs
                absolute_idx = step * len(args.skip_counts) * len(attention_modes) + mode_idx * len(args.skip_counts) + run_idx
                print(
                    f"[LLM {absolute_idx}/{total}] {phase} "
                    f"attn={str(output_attentions).lower()} skip={skip_count} "
                    f"seq={length_by_skip[int(skip_count)]}: cuda={timing['cuda_ms']:.3f} ms"
                )
                if step >= args.warmup:
                    rows.append(
                        {
                            "output_attentions": bool(output_attentions),
                            "skip_count": int(skip_count),
                            "seq_len": length_by_skip[int(skip_count)],
                            **timing,
                        }
                    )

    summary = summarize(rows, ["output_attentions", "skip_count", "seq_len"])
    baselines = {
        bool(item["output_attentions"]): float(item["cuda_ms_mean"])
        for item in summary
        if item["skip_count"] == 0
    }
    for item in summary:
        base = baselines.get(bool(item["output_attentions"]))
        if base and base > 0:
            current = float(item["cuda_ms_mean"])
            item["cuda_speedup_pct"] = (base - current) / base * 100.0
            item["cuda_speedup_x"] = base / current if current > 0 else 0.0
    return {
        "length_meta": length_meta,
        "attention_class": type(vla.language_model.model.layers[0].self_attn).__name__,
        "rows": rows,
        "summary": summary,
    }


def print_summary(name: str, summary: List[Dict[str, Any]]) -> None:
    print(f"\n{name} Summary")
    keys = [key for key in ("branch", "output_attentions", "skip_count", "seq_len") if key in summary[0]]
    print(" | ".join(keys + ["n", "cuda_mean", "cuda_median", "speedup_pct", "speedup_x"]))
    for item in summary:
        print(
            " | ".join(
                [str(item[key]) for key in keys]
                + [
                    str(item["n"]),
                    f"{item['cuda_ms_mean']:.3f}",
                    f"{item['cuda_ms_median']:.3f}",
                    f"{float(item.get('cuda_speedup_pct', 0.0)):.2f}",
                    f"{float(item.get('cuda_speedup_x', 1.0)):.3f}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = str(Path(args.checkpoint).resolve())
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Skip counts: {args.skip_counts}")

    old_cwd = Path.cwd()
    os.chdir(REPO_ROOT / "src" / "openvla")
    try:
        vla = get_vla(LoadConfig(checkpoint)).eval()
        processor = get_processor(LoadConfig(checkpoint))
    finally:
        os.chdir(old_cwd)

    payload: Dict[str, Any] = {
        "args": vars(args),
        "checkpoint": checkpoint,
        "device": str(DEVICE),
    }

    if args.bench in ("vit", "both"):
        vit = run_vit_oracle(vla, args)
        payload["vit"] = vit
        print_summary("ViT Oracle", vit["summary"])

    if args.bench in ("llm", "both"):
        llm = run_llm_oracle(vla, processor, args)
        payload["llm"] = llm
        print_summary("LLM Oracle", llm["summary"])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
