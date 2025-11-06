#!/usr/bin/env python3
"""
Utility script to compare ViT patch tokens under two strategies:

1) Baseline (TTF-style): run the full vision backbone on each frame, then swap
   a subset of patches with the *previous frame's fused tokens*.
2) Cache reuse: reuse the stored tokens via PrismaticVisionBackbone.set_reuse_mask.

The script prints per-frame max / mean absolute differences to verify that the
cache implementation reproduces the token-fusion baseline, including multi-frame
accumulation effects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image

from experiments.robot.robot_utils import get_model, set_seed_everywhere
from experiments.robot.openvla_utils import get_processor


@dataclass
class MinimalConfig:
    """Lightweight config stub to load the OpenVLA vision backbone."""

    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_vla_cache: bool = True
    use_vit_cache: bool = True


def build_config(checkpoint: str) -> MinimalConfig:
    cfg = MinimalConfig()
    cfg.pretrained_checkpoint = checkpoint
    return cfg


@torch.no_grad()
def build_test_images(num_frames: int, resolution: int = 224) -> List[Image.Image]:
    """Create a sequence of synthetic frames with localized changes."""
    rng = torch.Generator().manual_seed(0)
    base = torch.randint(
        low=40,
        high=80,
        size=(resolution, resolution, 3),
        generator=rng,
        dtype=torch.uint8,
    )
    frames = []
    for idx in range(num_frames):
        frame = base.clone()
        # Introduce a moving bright square so different patches become dynamic.
        start = 40 + 20 * idx
        end = start + 40
        frame[start:end, 120:160, :] = 200
        frames.append(Image.fromarray(frame.numpy()))
    return frames


def prepare_pixel_values(processor, prompt: str, image: Image.Image, device: torch.device) -> torch.Tensor:
    """Run OpenVLA processor to obtain pixel tensor."""
    batch = processor(text=[prompt], images=image, return_tensors="pt")
    return batch["pixel_values"].to(device=device, dtype=torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description="Compare ViT cache reuse against TTF token fusion.")
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        required=True,
        help="Path to the OpenVLA checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation.",
    )
    parser.add_argument(
        "--reuse_count",
        type=int,
        default=80,
        help="Number of patches (0-256) treated as static on each frame.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=3,
        help="Number of consecutive frames to evaluate (>=2).",
    )
    args = parser.parse_args()

    if args.reuse_count < 0 or args.reuse_count > 256:
        raise ValueError("--reuse_count must be between 0 and 256.")
    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2.")

    device = torch.device(args.device)
    set_seed_everywhere(42)

    cfg = build_config(args.pretrained_checkpoint)

    model = get_model(cfg)
    model.eval()
    model.to(device)
    processor = get_processor(cfg)
    backbone = model.vision_backbone

    frames = build_test_images(args.num_frames)
    prompt = "What action should the robot take?"
    pixels = [prepare_pixel_values(processor, prompt, frame, device) for frame in frames]

    # Collect full ViT outputs for each frame (no reuse) to form the TTF baseline.
    backbone.reset_cache()
    backbone.set_reuse_mask(None)
    full_tokens: List[torch.Tensor] = []
    for pixel in pixels:
        tokens = backbone(pixel).to(dtype=torch.float32).cpu()
        full_tokens.append(tokens)
    backbone.reset_cache()

    num_patches = full_tokens[0].shape[1]
    reuse_count = min(args.reuse_count, num_patches)

    rng = torch.Generator().manual_seed(1234)
    mask_sequence: List[Optional[torch.Tensor]] = [None]
    for _ in range(1, args.num_frames):
        if reuse_count == 0:
            mask_sequence.append(torch.zeros(num_patches, dtype=torch.bool))
            continue
        perm = torch.randperm(num_patches, generator=rng)
        mask = torch.zeros(num_patches, dtype=torch.bool)
        mask[perm[:reuse_count]] = True
        mask_sequence.append(mask)

    # Build TTF outputs with accumulated fusion.
    ttf_tokens: List[torch.Tensor] = []
    prev_fused = full_tokens[0]
    ttf_tokens.append(prev_fused)
    for idx in range(1, args.num_frames):
        fused = full_tokens[idx].clone()
        mask = mask_sequence[idx]
        if mask is not None and mask.any():
            fused[:, mask] = prev_fused[:, mask]
        prev_fused = fused
        ttf_tokens.append(fused)

    # Run cache path using the same masks.
    cache_tokens: List[torch.Tensor] = []
    backbone.reset_cache()
    backbone.set_reuse_mask(None)
    cache_tokens.append(backbone(pixels[0]).to(dtype=torch.float32).cpu())
    for idx in range(1, args.num_frames):
        backbone.set_reuse_mask(mask_sequence[idx])
        cache_tokens.append(backbone(pixels[idx]).to(dtype=torch.float32).cpu())
    backbone.set_reuse_mask(None)
    backbone.reset_cache()

    # Compare per frame.
    for idx, (ttf_tok, cache_tok, mask) in enumerate(zip(ttf_tokens, cache_tokens, mask_sequence)):
        diff = (ttf_tok - cache_tok).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        reused_max = 0.0
        new_max = 0.0
        if mask is not None and mask.any():
            reused_max = diff[:, mask].max().item()
            new_max = diff[:, ~mask].max().item()
        print(f"Frame {idx}: max diff={max_diff:.6f}, mean diff={mean_diff:.6f}, reused max={reused_max:.6f}, new max={new_max:.6f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
