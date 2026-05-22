#!/usr/bin/env python3
"""Bucketed CUDA Graph microbenchmark for OpenVLA LLM CRP.

This is a prototype execution backend, not the rollout path. It builds the
actual OpenVLA multimodal embeddings, a warmed previous-frame KV cache, and the
actual CRP masks from ``llm_crp_microbench.py``. Then it compares:

- full dense decoder on the full multimodal sequence;
- bucketed CRP decoder with reuse query rows removed and prune rows removed
  from the effective KV context;
- eager execution and fixed-shape CUDA Graph replay for both.

The goal is to test whether the CRP masks become latency-positive once the LLM
prefill is expressed as fixed-shape dense bucket computation.
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from llm_crp_microbench import (  # noqa: E402
    DEVICE,
    LoadConfig,
    append_action_prompt_token,
    build_masks,
    build_prompt,
    get_processor,
    get_vla,
    load_image,
    resolve_unnorm_key,
    tensor_numel,
    warm_reference_cache,
)


@dataclass
class BucketSpec:
    name: str
    query_positions: torch.Tensor
    kv_positions: torch.Tensor
    cache_position: torch.Tensor
    position_ids: torch.Tensor
    hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor]


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
    parser.add_argument("--image", default=None)
    parser.add_argument("--static-top-k", type=int, default=130)
    parser.add_argument("--attention-top-k", type=int, default=120)
    parser.add_argument("--prune-ratio", type=float, default=0.10)
    parser.add_argument("--target-reuse-count", type=int, default=None)
    parser.add_argument("--target-prune-count", type=int, default=None)
    parser.add_argument("--target-skip-count", type=int, default=None)
    parser.add_argument("--patch-metric", choices=("cosine", "gray_diff", "rgb_diff"), default="cosine")
    parser.add_argument("--sim-threshold", type=float, default=0.996)
    parser.add_argument("--diff-threshold", type=float, default=0.10)
    parser.add_argument("--output-attentions", action="store_true")
    parser.add_argument(
        "--mask-mode",
        choices=("position", "none"),
        default="position",
        help="Use an explicit original-position causal mask, or rely on SDPA is_causal when possible.",
    )
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


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


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["runner"], row["bucket"]), []).append(row)

    summary = []
    for (runner, bucket), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "runner": runner,
            "bucket": bucket,
            "n": len(group),
            "query_len": int(group[0]["query_len"]),
            "kv_len": int(group[0]["kv_len"]),
            "last_logit_checksum": float(group[-1].get("last_logit_checksum", 0.0)),
        }
        for metric in ("wall_ms", "cuda_ms"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_median"] = median(values)
        summary.append(item)

    dense_by_runner = {
        item["runner"]: float(item["cuda_ms_mean"])
        for item in summary
        if item["bucket"] == "dense"
    }
    eager_by_bucket = {
        item["bucket"]: float(item["cuda_ms_mean"])
        for item in summary
        if item["runner"] == "eager"
    }
    for item in summary:
        dense = dense_by_runner.get(item["runner"])
        if dense and dense > 0:
            current = float(item["cuda_ms_mean"])
            item["cuda_speedup_vs_dense_same_runner_pct"] = (dense - current) / dense * 100.0
            item["cuda_speedup_vs_dense_same_runner_x"] = dense / current if current > 0 else 0.0
        eager = eager_by_bucket.get(item["bucket"])
        if eager and eager > 0:
            current = float(item["cuda_ms_mean"])
            item["cuda_speedup_vs_eager_same_bucket_pct"] = (eager - current) / eager * 100.0
            item["cuda_speedup_vs_eager_same_bucket_x"] = eager / current if current > 0 else 0.0
    return summary


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\nBucketed CRP Graph Summary")
    print(
        "runner | bucket | q | kv | n | cuda_mean | cuda_median | "
        "speedup_vs_dense | speedup_vs_eager"
    )
    for item in summary:
        print(
            " | ".join(
                [
                    str(item["runner"]),
                    str(item["bucket"]),
                    str(item["query_len"]),
                    str(item["kv_len"]),
                    str(item["n"]),
                    f"{item['cuda_ms_mean']:.3f}",
                    f"{item['cuda_ms_median']:.3f}",
                    f"{float(item.get('cuda_speedup_vs_dense_same_runner_pct', 0.0)):.2f}%",
                    f"{float(item.get('cuda_speedup_vs_eager_same_bucket_pct', 0.0)):.2f}%",
                ]
            )
        )


class BucketCache:
    def __init__(self, key_cache: List[torch.Tensor], value_cache: List[torch.Tensor]) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs["cache_position"] if cache_kwargs is not None else None
        key_out = self.key_cache[layer_idx]
        value_out = self.value_cache[layer_idx]
        if cache_position is None:
            key_out.copy_(key_states)
            value_out.copy_(value_states)
        else:
            key_out.index_copy_(2, cache_position, key_states.to(key_out.dtype))
            value_out.index_copy_(2, cache_position, value_states.to(value_out.dtype))
        return key_out, value_out


def clone_bucket_cache(prompt_cache: Any, kv_positions: torch.Tensor) -> BucketCache:
    kv_positions = kv_positions.to(dtype=torch.long, device=DEVICE).flatten()
    key_cache = []
    value_cache = []
    for key_states, value_states in zip(prompt_cache.key_cache, prompt_cache.value_cache):
        key_cache.append(key_states.index_select(2, kv_positions).clone())
        value_cache.append(value_states.index_select(2, kv_positions).clone())
    return BucketCache(key_cache, value_cache)


def make_empty_like_cache(base_cache: BucketCache) -> BucketCache:
    return BucketCache(
        [torch.empty_like(tensor) for tensor in base_cache.key_cache],
        [torch.empty_like(tensor) for tensor in base_cache.value_cache],
    )


def copy_cache_(dst: BucketCache, src: BucketCache) -> None:
    for dst_key, src_key in zip(dst.key_cache, src.key_cache):
        dst_key.copy_(src_key)
    for dst_value, src_value in zip(dst.value_cache, src.value_cache):
        dst_value.copy_(src_value)


def build_position_mask(
    query_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    min_dtype = torch.finfo(dtype).min
    allowed = kv_positions[None, :] <= query_positions[:, None]
    mask = torch.zeros(query_positions.numel(), kv_positions.numel(), dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, min_dtype)
    return mask[None, None, :, :]


@torch.inference_mode()
def build_multimodal_embeddings(vla: Any, inputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = append_action_prompt_token(inputs["input_ids"])
    projected = vla.projector(vla.vision_backbone(inputs["pixel_values"]))
    input_embeddings = vla.get_input_embeddings()(input_ids)
    multimodal_embeddings = torch.cat(
        [input_embeddings[:, :1, :], projected, input_embeddings[:, 1:, :]],
        dim=1,
    )
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        multimodal_attention_mask = torch.ones(
            multimodal_embeddings.shape[:2],
            dtype=torch.long,
            device=multimodal_embeddings.device,
        )
    else:
        projected_mask = torch.ones(
            projected.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], projected_mask, attention_mask[:, 1:]],
            dim=1,
        )
    return multimodal_embeddings, multimodal_attention_mask


def build_bucket_specs(
    embeddings: torch.Tensor,
    reusable: torch.Tensor,
    pruned: torch.Tensor,
    output_attentions: bool,
    mask_mode: str,
) -> Dict[str, BucketSpec]:
    del output_attentions
    total_len = int(embeddings.shape[1])
    full_positions = torch.arange(total_len, dtype=torch.long, device=embeddings.device)
    reuse_or_prune = torch.unique(torch.cat([reusable, pruned], dim=0), sorted=True)
    reuse_or_prune = reuse_or_prune[(reuse_or_prune >= 0) & (reuse_or_prune < total_len)]
    pruned = torch.unique(pruned, sorted=True)
    pruned = pruned[(pruned >= 0) & (pruned < total_len)]

    query_visible = torch.ones(total_len, dtype=torch.bool, device=embeddings.device)
    if reuse_or_prune.numel() > 0:
        query_visible[reuse_or_prune] = False
    kv_visible = torch.ones(total_len, dtype=torch.bool, device=embeddings.device)
    if pruned.numel() > 0:
        kv_visible[pruned] = False

    crp_query_positions = full_positions[query_visible]
    crp_kv_positions = full_positions[kv_visible]
    original_to_compact = torch.full((total_len,), -1, dtype=torch.long, device=embeddings.device)
    original_to_compact[crp_kv_positions] = torch.arange(
        crp_kv_positions.numel(), dtype=torch.long, device=embeddings.device
    )

    def _make(name: str, query_positions: torch.Tensor, kv_positions: torch.Tensor) -> BucketSpec:
        hidden_states = embeddings.index_select(1, query_positions).contiguous()
        position_ids = query_positions.unsqueeze(0).contiguous()
        cache_position = torch.arange(kv_positions.numel(), device=embeddings.device, dtype=torch.long)
        if name == "crp":
            cache_position = original_to_compact.index_select(0, query_positions).contiguous()
        attention_mask = None
        if mask_mode == "position":
            attention_mask = build_position_mask(
                query_positions,
                kv_positions,
                embeddings.dtype,
                embeddings.device,
            )
        return BucketSpec(
            name=name,
            query_positions=query_positions,
            kv_positions=kv_positions,
            cache_position=cache_position,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    return {
        "dense": _make("dense", full_positions, full_positions),
        "crp": _make("crp", crp_query_positions, crp_kv_positions),
    }


@torch.inference_mode()
def run_bucket_decoder(
    vla: Any,
    hidden_states: torch.Tensor,
    cache: BucketCache,
    spec: BucketSpec,
    output_attentions: bool,
) -> torch.Tensor:
    model = vla.language_model.model
    for decoder_layer in model.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=spec.attention_mask,
            position_ids=spec.position_ids,
            past_key_value=cache,
            output_attentions=output_attentions,
            use_cache=True,
            cache_position=spec.cache_position,
        )
        hidden_states = layer_outputs[0]
    hidden_states = model.norm(hidden_states)
    logits = vla.language_model.lm_head(hidden_states)
    return logits


class GraphRunner:
    def __init__(
        self,
        vla: Any,
        spec: BucketSpec,
        base_cache: BucketCache,
        output_attentions: bool,
        graph_warmup: int,
    ) -> None:
        self.vla = vla
        self.spec = spec
        self.output_attentions = output_attentions
        self.static_hidden = torch.empty_like(spec.hidden_states)
        self.cache = make_empty_like_cache(base_cache)
        self.graph = torch.cuda.CUDAGraph()
        self.output: Optional[torch.Tensor] = None

        self.copy_inputs_(spec.hidden_states, base_cache)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(graph_warmup):
                copy_cache_(self.cache, base_cache)
                self.static_hidden.copy_(spec.hidden_states)
                self.output = run_bucket_decoder(
                    self.vla,
                    self.static_hidden,
                    self.cache,
                    self.spec,
                    self.output_attentions,
                )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        copy_cache_(self.cache, base_cache)
        self.static_hidden.copy_(spec.hidden_states)
        with torch.cuda.graph(self.graph):
            self.output = run_bucket_decoder(
                self.vla,
                self.static_hidden,
                self.cache,
                self.spec,
                self.output_attentions,
            )

    def copy_inputs_(self, hidden_states: torch.Tensor, cache: BucketCache) -> None:
        self.static_hidden.copy_(hidden_states)
        copy_cache_(self.cache, cache)

    def replay(self, hidden_states: torch.Tensor, cache: BucketCache, include_copy: bool) -> torch.Tensor:
        if include_copy:
            self.copy_inputs_(hidden_states, cache)
        self.graph.replay()
        assert self.output is not None
        return self.output


def logit_checksum(logits: torch.Tensor) -> float:
    last = logits[:, -1, :]
    return float(last.float().mean().detach().cpu())


@torch.inference_mode()
def validate_graph_outputs(
    vla: Any,
    specs: Dict[str, BucketSpec],
    base_caches: Dict[str, BucketCache],
    graph_runners: Dict[str, GraphRunner],
    output_attentions: bool,
) -> List[Dict[str, Any]]:
    validations = []
    if not graph_runners:
        return validations
    for name, spec in specs.items():
        eager_cache = make_empty_like_cache(base_caches[name])
        copy_cache_(eager_cache, base_caches[name])
        eager_logits = run_bucket_decoder(vla, spec.hidden_states, eager_cache, spec, output_attentions)
        graph_logits = graph_runners[name].replay(spec.hidden_states, base_caches[name], include_copy=True)
        diff = (eager_logits.float() - graph_logits.float()).abs()
        validations.append(
            {
                "bucket": name,
                "max_abs_diff": float(diff.max().detach().cpu()),
                "mean_abs_diff": float(diff.mean().detach().cpu()),
                "eager_last_logit_checksum": logit_checksum(eager_logits),
                "graph_last_logit_checksum": logit_checksum(graph_logits),
            }
        )
        del eager_logits, graph_logits, diff
    return validations


@torch.inference_mode()
def run_benchmark(vla: Any, prompt_cache: Any, specs: Dict[str, BucketSpec], args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    base_caches = {
        name: clone_bucket_cache(prompt_cache, spec.kv_positions)
        for name, spec in specs.items()
    }
    work_caches = {
        name: make_empty_like_cache(cache)
        for name, cache in base_caches.items()
    }

    graph_runners = {}
    if torch.cuda.is_available():
        for name, spec in specs.items():
            graph_runners[name] = GraphRunner(
                vla,
                spec,
                base_caches[name],
                args.output_attentions,
                args.graph_warmup,
            )

    validations = validate_graph_outputs(
        vla,
        specs,
        base_caches,
        graph_runners,
        args.output_attentions,
    )
    for item in validations:
        print(
            f"Validation {item['bucket']}: graph_vs_eager "
            f"max_abs={item['max_abs_diff']:.6f} mean_abs={item['mean_abs_diff']:.6f}"
        )

    total = (args.warmup + args.iters) * len(specs) * (4 if torch.cuda.is_available() else 2)
    run_idx = 0
    for step in range(args.warmup + args.iters):
        phase = "warmup" if step < args.warmup else "measure"
        for name, spec in specs.items():
            base_cache = base_caches[name]
            work_cache = work_caches[name]

            copy_cache_(work_cache, base_cache)
            wall_start, start_event = timer_start()
            logits = run_bucket_decoder(vla, spec.hidden_states, work_cache, spec, args.output_attentions)
            timing = timer_stop(wall_start, start_event)
            checksum = logit_checksum(logits)
            del logits
            run_idx += 1
            print(
                f"[{run_idx}/{total}] {phase} eager {name}: "
                f"q={spec.query_positions.numel()} kv={spec.kv_positions.numel()} cuda={timing['cuda_ms']:.3f} ms"
            )
            if step >= args.warmup:
                rows.append(
                    {
                        "runner": "eager",
                        "bucket": name,
                        "query_len": int(spec.query_positions.numel()),
                        "kv_len": int(spec.kv_positions.numel()),
                        "last_logit_checksum": checksum,
                        **timing,
                    }
                )

            wall_start, start_event = timer_start()
            copy_cache_(work_cache, base_cache)
            logits = run_bucket_decoder(vla, spec.hidden_states, work_cache, spec, args.output_attentions)
            copy_timing = timer_stop(wall_start, start_event)
            checksum = logit_checksum(logits)
            del logits
            run_idx += 1
            print(
                f"[{run_idx}/{total}] {phase} copy+eager {name}: "
                f"q={spec.query_positions.numel()} kv={spec.kv_positions.numel()} cuda={copy_timing['cuda_ms']:.3f} ms"
            )
            if step >= args.warmup:
                rows.append(
                    {
                        "runner": "copy_eager",
                        "bucket": name,
                        "query_len": int(spec.query_positions.numel()),
                        "kv_len": int(spec.kv_positions.numel()),
                        "last_logit_checksum": checksum,
                        **copy_timing,
                    }
                )

            if not torch.cuda.is_available():
                continue

            runner = graph_runners[name]
            wall_start, start_event = timer_start()
            logits = runner.replay(spec.hidden_states, base_cache, include_copy=False)
            graph_timing = timer_stop(wall_start, start_event)
            checksum = logit_checksum(logits)
            run_idx += 1
            print(
                f"[{run_idx}/{total}] {phase} graph {name}: "
                f"q={spec.query_positions.numel()} kv={spec.kv_positions.numel()} cuda={graph_timing['cuda_ms']:.3f} ms"
            )
            if step >= args.warmup:
                rows.append(
                    {
                        "runner": "graph",
                        "bucket": name,
                        "query_len": int(spec.query_positions.numel()),
                        "kv_len": int(spec.kv_positions.numel()),
                        "last_logit_checksum": checksum,
                        **graph_timing,
                    }
                )

            wall_start, start_event = timer_start()
            logits = runner.replay(spec.hidden_states, base_cache, include_copy=True)
            copy_graph_timing = timer_stop(wall_start, start_event)
            checksum = logit_checksum(logits)
            run_idx += 1
            print(
                f"[{run_idx}/{total}] {phase} copy+graph {name}: "
                f"q={spec.query_positions.numel()} kv={spec.kv_positions.numel()} cuda={copy_graph_timing['cuda_ms']:.3f} ms"
            )
            if step >= args.warmup:
                rows.append(
                    {
                        "runner": "copy_graph",
                        "bucket": name,
                        "query_len": int(spec.query_positions.numel()),
                        "kv_len": int(spec.kv_positions.numel()),
                        "last_logit_checksum": checksum,
                        **copy_graph_timing,
                    }
                )

    summary = summarize(rows)
    return {"validations": validations, "rows": rows, "summary": summary}


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
    print(f"Output attentions: {args.output_attentions}")
    print(f"Mask mode: {args.mask_mode}")

    cfg = LoadConfig(pretrained_checkpoint=checkpoint)
    old_cwd = Path.cwd()
    os.chdir(REPO_ROOT / "src" / "openvla")
    try:
        vla = get_vla(cfg).eval()
        processor = get_processor(cfg)
    finally:
        os.chdir(old_cwd)

    unnorm_key = resolve_unnorm_key(vla, args.task_suite_name)
    image = load_image(args.image, args.seed)
    prompt = build_prompt(checkpoint, args.task_label)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    print(f"Prompt tokens: {inputs['input_ids'].shape[-1]}")
    print(f"Unnorm key: {unnorm_key}")
    print(f"LLM attention class: {type(vla.language_model.model.layers[0].self_attn).__name__}")

    print("Warming reference cache and attention masks...")
    reference = warm_reference_cache(vla, inputs, unnorm_key)
    prompt_cache = reference["past_key_values"]
    masks = build_masks(reference["attentions"], image, args)
    reusable = masks["reusable_positions"]
    pruned = masks["pruned_positions"]
    print(
        "Mask stats: "
        f"stable={masks['stable_patch_count']} reuse={tensor_numel(reusable)} "
        f"prune={tensor_numel(pruned)} fill={masks['target_fill_count']}"
    )

    embeddings, multimodal_attention_mask = build_multimodal_embeddings(vla, inputs)
    del multimodal_attention_mask
    specs = build_bucket_specs(
        embeddings,
        reusable,
        pruned,
        output_attentions=args.output_attentions,
        mask_mode=args.mask_mode,
    )
    for spec in specs.values():
        print(
            f"Bucket {spec.name}: q={spec.query_positions.numel()} kv={spec.kv_positions.numel()} "
            f"last_q_pos={int(spec.query_positions[-1].item())}"
        )

    bench = run_benchmark(vla, prompt_cache, specs, args)
    print_summary(bench["summary"])

    payload = {
        "args": vars(args),
        "checkpoint": checkpoint,
        "device": str(DEVICE),
        "unnorm_key": unnorm_key,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "attention_class": type(vla.language_model.model.layers[0].self_attn).__name__,
        "mask_stats": {
            "stable_patch_count": masks["stable_patch_count"],
            "target_fill_count": masks["target_fill_count"],
            "reuse_count": tensor_numel(reusable),
            "prune_count": tensor_numel(pruned),
        },
        "buckets": {
            name: {
                "query_len": int(spec.query_positions.numel()),
                "kv_len": int(spec.kv_positions.numel()),
            }
            for name, spec in specs.items()
        },
        **bench,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
