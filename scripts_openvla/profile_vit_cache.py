#!/usr/bin/env python3
"""
Minimal profiler to inspect ViT cache reuse hotspots.

Runs two forwards on a timm ViT:
  - first pass fills caches
  - second pass profiles reuse path and dumps a Chrome trace + top ops

Usage: python scripts/profile_vit_cache.py [--model vit_large_patch14_reg4_dinov2]
"""
import argparse
import os
import torch
import timm
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_large_patch14_reg4_dinov2")
    parser.add_argument("--img-size", type=int, default=None, help="override image size; default uses model cfg")
    parser.add_argument("--static-ratio", type=float, default=0.3, help="portion of patch tokens marked static")
    parser.add_argument("--trace-dir", default="results/vit_profile_trace")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = timm.create_model(args.model, pretrained=False)
    # Resolve image size from model default if not provided
    if args.img_size is None:
        cfg_input = model.default_cfg.get("input_size", (3, 224, 224))
        img_size = int(cfg_input[1])
    else:
        img_size = args.img_size
    # Move to device/dtype after resolving img_size
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Build reuse mask: prefix tokens + patch tokens
    num_prefix = getattr(model, "num_prefix_tokens", 0)
    num_patches = model.patch_embed.num_patches
    reuse_mask = torch.zeros(num_prefix + num_patches, dtype=torch.bool, device=device)
    static_patches = int(num_patches * args.static_ratio)
    if static_patches > 0:
        reuse_mask[num_prefix : num_prefix + static_patches] = True

    # Attach empty caches; first pass will populate
    cache_state = [None] * len(model.blocks)
    model.set_vla_cache_state(cache_state, reuse_mask, enable_reuse=True)

    # Dummy input
    x = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)

    # Warmup: fill cache
    with torch.no_grad():
        model(x)

    # Profiling second forward (reuse path)
    trace_dir = args.trace_dir
    os.makedirs(trace_dir, exist_ok=True)
    with torch.no_grad(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(trace_dir),
    ) as prof:
        model(x)

    print(f"Trace written to {trace_dir} (load with TensorBoard chrome trace). Top ops:")
    ka = prof.key_averages(group_by_input_shape=True)
    if device == "cuda":
        print("== Sorted by cuda_time_total ==")
        print(ka.table(sort_by="cuda_time_total", row_limit=50))
        print("\n== Sorted by self_cuda_time_total ==")
        print(ka.table(sort_by="self_cuda_time_total", row_limit=50))
    else:
        print("== Sorted by cpu_time_total ==")
        print(ka.table(sort_by="cpu_time_total", row_limit=50))
        print("\n== Sorted by self_cpu_time_total ==")
        print(ka.table(sort_by="self_cpu_time_total", row_limit=50))


if __name__ == "__main__":
    main()
