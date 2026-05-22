#!/usr/bin/env python3
"""CUDA Graph oracle for short dense LLM prefill.

This benchmark does not exercise the CRP implementation. It answers whether
the corrected dense-short-sequence LLM oracle is mostly limited by launch and
dispatch gaps. For each sequence length it compares:

- normal: direct decoder execution with precomputed static args;
- cuda_graph_replay: replay of a captured fixed-shape graph;
- cuda_graph_copy_replay: copy into the graph input buffer plus replay.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from oracle_crp_microbench import (  # noqa: E402
    DEVICE,
    LoadConfig,
    build_4d_causal_mask,
    get_processor,
    get_vla,
    infer_multimodal_length,
    reset_llm_crp,
    timer_start,
    timer_stop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "src" / "openvla" / "checkpoints" / "openvla-7b-finetuned-libero-spatial"),
    )
    parser.add_argument("--task-label", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--skip-counts", type=int, nargs="+", default=[0, 75, 128, 192, 224, 250])
    parser.add_argument(
        "--llm-attention-modes",
        choices=("false", "true", "both"),
        default="false",
        help="Whether to benchmark output_attentions disabled, enabled, or both.",
    )
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["runner"],
            row["output_attentions"],
            row["skip_count"],
            row["seq_len"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in sorted(groups.items()):
        item = {
            "runner": key[0],
            "output_attentions": key[1],
            "skip_count": key[2],
            "seq_len": key[3],
            "n": len(group),
        }
        for metric in ("wall_ms", "cuda_ms"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_median"] = median(values)
        summary.append(item)

    baselines: Dict[tuple[str, bool], float] = {}
    normal_by_shape: Dict[tuple[bool, int], float] = {}
    for item in summary:
        if item["skip_count"] == 0:
            baselines[(str(item["runner"]), bool(item["output_attentions"]))] = float(item["cuda_ms_mean"])
        if item["runner"] == "normal":
            normal_by_shape[(bool(item["output_attentions"]), int(item["seq_len"]))] = float(item["cuda_ms_mean"])

    for item in summary:
        base = baselines.get((str(item["runner"]), bool(item["output_attentions"])))
        if base and base > 0:
            current = float(item["cuda_ms_mean"])
            item["cuda_speedup_vs_runner_skip0_pct"] = (base - current) / base * 100.0
            item["cuda_speedup_vs_runner_skip0_x"] = base / current if current > 0 else 0.0
        normal = normal_by_shape.get((bool(item["output_attentions"]), int(item["seq_len"])))
        if normal and normal > 0:
            current = float(item["cuda_ms_mean"])
            item["cuda_speedup_vs_normal_same_seq_pct"] = (normal - current) / normal * 100.0
            item["cuda_speedup_vs_normal_same_seq_x"] = normal / current if current > 0 else 0.0
    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\nLLM CUDA Graph Oracle Summary")
    print(
        "runner | attn | skip | seq | n | cuda_mean | cuda_median | "
        "speedup_vs_own_skip0 | speedup_vs_normal_same_seq"
    )
    for item in summary:
        print(
            " | ".join(
                [
                    str(item["runner"]),
                    str(item["output_attentions"]).lower(),
                    str(item["skip_count"]),
                    str(item["seq_len"]),
                    str(item["n"]),
                    f"{item['cuda_ms_mean']:.3f}",
                    f"{item['cuda_ms_median']:.3f}",
                    f"{float(item.get('cuda_speedup_vs_runner_skip0_pct', 0.0)):.2f}%",
                    f"{float(item.get('cuda_speedup_vs_normal_same_seq_pct', 0.0)):.2f}%",
                ]
            )
        )


@torch.inference_mode()
def run_llm_decoder_static(
    vla: Any,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    output_attentions: bool,
) -> torch.Tensor:
    model = vla.language_model.model
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


class CudaGraphRunner:
    def __init__(
        self,
        vla: Any,
        static_input: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        output_attentions: bool,
        graph_warmup: int,
    ) -> None:
        self.vla = vla
        self.static_input = static_input
        self.position_ids = position_ids
        self.cache_position = cache_position
        self.causal_mask = causal_mask
        self.output_attentions = output_attentions
        self.graph = torch.cuda.CUDAGraph()
        self.output: Optional[torch.Tensor] = None

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(graph_warmup):
                self.output = run_llm_decoder_static(
                    self.vla,
                    self.static_input,
                    self.position_ids,
                    self.cache_position,
                    self.causal_mask,
                    self.output_attentions,
                )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        with torch.cuda.graph(self.graph):
            self.output = run_llm_decoder_static(
                self.vla,
                self.static_input,
                self.position_ids,
                self.cache_position,
                self.causal_mask,
                self.output_attentions,
            )

    def replay(self, source: torch.Tensor, include_copy: bool) -> torch.Tensor:
        if include_copy:
            self.static_input.copy_(source)
        self.graph.replay()
        assert self.output is not None
        return self.output


def make_inputs(
    length_meta: Dict[str, int],
    hidden_size: int,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[Dict[int, int], Dict[int, torch.Tensor]]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(args.seed + 3000)
    seq_by_skip: Dict[int, int] = {}
    tensors: Dict[int, torch.Tensor] = {}
    full_len = int(length_meta["full_seq_len"])
    vision_tokens = int(length_meta["vision_tokens"])
    for skip_count in args.skip_counts:
        effective_skip = max(0, min(int(skip_count), vision_tokens))
        seq_len = full_len - effective_skip
        seq_by_skip[int(skip_count)] = seq_len
        tensors[int(skip_count)] = torch.randn(
            1,
            seq_len,
            hidden_size,
            device=DEVICE,
            dtype=dtype,
            generator=generator,
        )
    return seq_by_skip, tensors


def build_static_args(seq_len: int, dtype: torch.dtype, output_attentions: bool) -> Dict[str, Optional[torch.Tensor]]:
    position_ids = torch.arange(seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0)
    cache_position = position_ids.squeeze(0)
    causal_mask = build_4d_causal_mask(seq_len, dtype, DEVICE) if output_attentions else None
    return {
        "position_ids": position_ids,
        "cache_position": cache_position,
        "causal_mask": causal_mask,
    }


def attention_modes(args: argparse.Namespace) -> List[bool]:
    if args.llm_attention_modes == "both":
        return [False, True]
    return [args.llm_attention_modes == "true"]


@torch.inference_mode()
def run_benchmark(vla: Any, processor: Any, args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Graph benchmark requires CUDA.")

    reset_llm_crp(vla)
    length_meta = infer_multimodal_length(vla, processor, args)
    hidden_size = int(vla.language_model.config.hidden_size)
    dtype = next(vla.language_model.parameters()).dtype
    seq_by_skip, tensors = make_inputs(length_meta, hidden_size, dtype, args)

    rows: List[Dict[str, Any]] = []
    graph_errors: List[Dict[str, Any]] = []
    modes = attention_modes(args)

    total = (args.warmup + args.iters) * len(args.skip_counts) * len(modes)
    for step in range(args.warmup + args.iters):
        for output_attentions in modes:
            for skip_idx, skip_count in enumerate(args.skip_counts, start=1):
                seq_len = seq_by_skip[int(skip_count)]
                static_args = build_static_args(seq_len, dtype, output_attentions)
                source = tensors[int(skip_count)]
                phase = "warmup" if step < args.warmup else "measure"
                absolute_idx = step * len(args.skip_counts) * len(modes) + skip_idx

                wall_start, start_event = timer_start()
                output = run_llm_decoder_static(
                    vla,
                    source,
                    static_args["position_ids"],  # type: ignore[arg-type]
                    static_args["cache_position"],  # type: ignore[arg-type]
                    static_args["causal_mask"],
                    output_attentions,
                )
                normal_timing = timer_stop(wall_start, start_event)
                del output
                print(
                    f"[normal {absolute_idx}/{total}] {phase} "
                    f"attn={str(output_attentions).lower()} skip={skip_count} seq={seq_len}: "
                    f"cuda={normal_timing['cuda_ms']:.3f} ms"
                )
                if step >= args.warmup:
                    rows.append(
                        {
                            "runner": "normal",
                            "output_attentions": bool(output_attentions),
                            "skip_count": int(skip_count),
                            "seq_len": int(seq_len),
                            **normal_timing,
                        }
                    )

    graph_runners: Dict[tuple[bool, int], CudaGraphRunner] = {}
    for output_attentions in modes:
        for skip_count in args.skip_counts:
            seq_len = seq_by_skip[int(skip_count)]
            static_args = build_static_args(seq_len, dtype, output_attentions)
            static_input = torch.empty_like(tensors[int(skip_count)])
            static_input.copy_(tensors[int(skip_count)])
            try:
                graph_runners[(bool(output_attentions), int(skip_count))] = CudaGraphRunner(
                    vla,
                    static_input,
                    static_args["position_ids"],  # type: ignore[arg-type]
                    static_args["cache_position"],  # type: ignore[arg-type]
                    static_args["causal_mask"],
                    output_attentions,
                    graph_warmup=args.graph_warmup,
                )
            except Exception as exc:  # noqa: BLE001 - keep benchmark running across modes
                error = {
                    "output_attentions": bool(output_attentions),
                    "skip_count": int(skip_count),
                    "seq_len": int(seq_len),
                    "error": repr(exc),
                }
                graph_errors.append(error)
                print(f"[graph capture failed] {error}")
            torch.cuda.synchronize()

    for step in range(args.warmup + args.iters):
        for output_attentions in modes:
            for skip_idx, skip_count in enumerate(args.skip_counts, start=1):
                runner = graph_runners.get((bool(output_attentions), int(skip_count)))
                if runner is None:
                    continue
                seq_len = seq_by_skip[int(skip_count)]
                source = tensors[int(skip_count)]
                phase = "warmup" if step < args.warmup else "measure"
                absolute_idx = step * len(args.skip_counts) * len(modes) + skip_idx

                wall_start, start_event = timer_start()
                output = runner.replay(source, include_copy=False)
                graph_timing = timer_stop(wall_start, start_event)
                del output
                print(
                    f"[graph {absolute_idx}/{total}] {phase} "
                    f"attn={str(output_attentions).lower()} skip={skip_count} seq={seq_len}: "
                    f"cuda={graph_timing['cuda_ms']:.3f} ms"
                )
                if step >= args.warmup:
                    rows.append(
                        {
                            "runner": "cuda_graph_replay",
                            "output_attentions": bool(output_attentions),
                            "skip_count": int(skip_count),
                            "seq_len": int(seq_len),
                            **graph_timing,
                        }
                    )

                wall_start, start_event = timer_start()
                output = runner.replay(source, include_copy=True)
                copy_graph_timing = timer_stop(wall_start, start_event)
                del output
                print(
                    f"[copy+graph {absolute_idx}/{total}] {phase} "
                    f"attn={str(output_attentions).lower()} skip={skip_count} seq={seq_len}: "
                    f"cuda={copy_graph_timing['cuda_ms']:.3f} ms"
                )
                if step >= args.warmup:
                    rows.append(
                        {
                            "runner": "cuda_graph_copy_replay",
                            "output_attentions": bool(output_attentions),
                            "skip_count": int(skip_count),
                            "seq_len": int(seq_len),
                            **copy_graph_timing,
                        }
                    )

    summary = summarize(rows)
    return {
        "length_meta": length_meta,
        "attention_class": type(vla.language_model.model.layers[0].self_attn).__name__,
        "graph_errors": graph_errors,
        "rows": rows,
        "summary": summary,
    }


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
        "llm": run_benchmark(vla, processor, args),
    }
    print_summary(payload["llm"]["summary"])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
