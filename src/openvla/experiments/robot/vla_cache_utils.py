import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.util import view_as_blocks


@torch.no_grad()
def get_layer_mask_schedule(multihead_attention, apply_weighted_growth=True, growth_factor=0.55):
    """
    Computes per-layer reuse proportions based on normalized attention entropy.

    Args:
        multihead_attention (List[Tensor]): Attention maps per layer (shape: [1, heads, tokens, tokens]).
        apply_weighted_growth (bool): Whether to smooth upward deltas.
        growth_factor (float): Weight for smoothing.

    Returns:
        torch.Tensor: Layer-wise reuse proportions, shape (num_layers - 1,).
    """
    device = multihead_attention[0].device
    entropies = []
    
    for attn in multihead_attention[:-1]:
        attn = attn.mean(dim=1)[0]
        attn /= attn.sum(dim=-1, keepdim=True) + 1e-10
        attn = torch.nan_to_num(attn, nan=0.0)
        token_entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)
        entropies.append(token_entropy.mean())

    entropies = torch.stack(entropies)
    norm_entropy = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
    reuse = 1.0 - norm_entropy

    if apply_weighted_growth:
        reuse = reuse.tolist()
        for i in range(1, len(reuse)):
            delta = reuse[i] - reuse[i - 1]
            if delta > 0:
                reuse[i] = reuse[i - 1] + delta * growth_factor
        reuse = torch.tensor(reuse, dtype=torch.float32, device=device)

    return reuse

def patchify(image, patch_size=14):
    """
    Converts an image into non-overlapping patches.
    """
    image = np.array(image)
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0, "Image dimensions must be divisible by patch size."

    if image.ndim == 3:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size, image.shape[2]))
    else:
        blocks = view_as_blocks(image, block_shape=(patch_size, patch_size))

    patches = blocks.reshape(-1, patch_size, patch_size, image.shape[2]) if image.ndim == 3 else blocks.reshape(-1, patch_size, patch_size)
    return patches

def calculate_patch_similarity(patches1, patches2):
    """
    Computes cosine similarity between two sets of patches.
    """
    flat1 = patches1.reshape(len(patches1), -1).astype(np.float32)
    flat2 = patches2.reshape(len(patches2), -1).astype(np.float32)
    
    norm1 = np.linalg.norm(flat1, axis=1)
    norm2 = np.linalg.norm(flat2, axis=1)
    
    dot = np.sum(flat1 * flat2, axis=1)
    cosine_sim = dot / (norm1 * norm2 + 1e-8)
    return cosine_sim


def calculate_patch_diff_gray(patches1, patches2):
    """
    Computes mean absolute grayscale difference per patch (0-1 scale).
    """
    p1 = patches1.astype(np.float32) / 255.0
    p2 = patches2.astype(np.float32) / 255.0
    if p1.ndim == 4:
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        gray1 = np.tensordot(p1, weights, axes=([-1], [0]))
        gray2 = np.tensordot(p2, weights, axes=([-1], [0]))
    else:
        gray1 = p1
        gray2 = p2
    diff = np.abs(gray1 - gray2)
    return diff.mean(axis=(-1, -2))


def calculate_patch_diff_rgb(patches1, patches2):
    """
    Computes mean absolute RGB difference per patch (0-1 scale).
    """
    p1 = patches1.astype(np.float32) / 255.0
    p2 = patches2.astype(np.float32) / 255.0
    diff = np.abs(p1 - p2)
    return diff.mean(axis=(1, 2, 3))

def find_static_patches(
    img_0,
    img_1,
    patch_size=14,
    top_k=150,
    sim_threshold=0.996,
    metric="cosine",
    diff_threshold=0.03,
):
    """
    Identifies significant patches with high similarity across two images.
    """
    patches1 = patchify(img_0, patch_size)
    patches2 = patchify(img_1, patch_size)

    grid_size = 224 // patch_size
    metric = metric.lower()

    if metric == "cosine":
        similarity = calculate_patch_similarity(patches1, patches2)
        similarity_2d = similarity.reshape(grid_size, grid_size)
        patch_scores = [
            (i * grid_size + j, similarity_2d[i, j])
            for i in range(grid_size)
            for j in range(grid_size)
            if similarity_2d[i, j] >= sim_threshold
        ]
        patch_scores.sort(key=lambda x: x[1], reverse=True)
    elif metric == "gray_diff":
        diff = calculate_patch_diff_gray(patches1, patches2).reshape(grid_size, grid_size)
        patch_scores = [
            (i * grid_size + j, diff[i, j])
            for i in range(grid_size)
            for j in range(grid_size)
            if diff[i, j] <= diff_threshold
        ]
        patch_scores.sort(key=lambda x: x[1])
    elif metric == "rgb_diff":
        diff = calculate_patch_diff_rgb(patches1, patches2).reshape(grid_size, grid_size)
        patch_scores = [
            (i * grid_size + j, diff[i, j])
            for i in range(grid_size)
            for j in range(grid_size)
            if diff[i, j] <= diff_threshold
        ]
        patch_scores.sort(key=lambda x: x[1])
    else:
        raise ValueError(f"Unknown patch metric: {metric}")

    top_patch_ids = [idx for idx, _ in patch_scores[:top_k]]
    return top_patch_ids

@torch.no_grad()
def token_attention_merge(multihead_attention, layer_id=15):
    """
    Computes mean attention from text tokens to vision tokens.
    """
    attn_map = multihead_attention[layer_id].to(torch.float32).squeeze(0).mean(dim=0)

    v_token_start = 1
    v_token_end = v_token_start + 256
    t_token_start = v_token_end
    t_token_end = t_token_start + 35

    position_meta = multihead_attention[-1]
    if isinstance(position_meta, (tuple, list)):
        query_pos, key_pos = position_meta
    else:
        query_pos = key_pos = position_meta
    query_pos = query_pos.to(attn_map.device)
    key_pos = key_pos.to(attn_map.device)

    text_mask = (query_pos >= t_token_start) & (query_pos < t_token_end)
    vision_mask = (key_pos >= v_token_start) & (key_pos < v_token_end)
    scores = torch.zeros(256, dtype=torch.float32, device=attn_map.device)
    if text_mask.any() and vision_mask.any():
        relation = attn_map[text_mask][:, vision_mask].mean(dim=0)
        patch_ids = (key_pos[vision_mask] - v_token_start).to(torch.long)
        valid = (patch_ids >= 0) & (patch_ids < 256)
        scores[patch_ids[valid]] = relation[valid]
    return scores.cpu()

def get_top_attention_patches(attn_scores, top_k=120):
    """
    Selects top-k patch indices based on attention scores.
    """
    attn_scores = attn_scores.cpu().numpy() if isinstance(attn_scores, torch.Tensor) else attn_scores
    attn_scores = np.asarray(attn_scores, dtype=np.float32).reshape(-1)
    if attn_scores.size < 256:
        attn_scores = np.pad(attn_scores, (0, 256 - attn_scores.size), constant_values=0.0)
    elif attn_scores.size > 256:
        attn_scores = attn_scores[:256]
    attn = attn_scores.reshape(16, 16)
    attn_resized = cv2.resize(attn, (16, 16))

    flat = [(i * 16 + j, attn_resized[i, j]) for i in range(16) for j in range(16)]
    flat.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in flat[:top_k]]

def draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4, draw_grid=True):
    """
    Draws colored overlays on image for different patch groups.
    """
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width = image.size[0]
    num_patches = width // patch_size

    if draw_grid:
        grid_color = (255, 255, 255, int(255 * 0.25))
        for i in range(0, width + 1, patch_size):
            draw.line([(i, 0), (i, width)], fill=grid_color, width=1)
            draw.line([(0, i), (width, i)], fill=grid_color, width=1)

    for patch_list, color in patch_groups:
        for pid in patch_list:
            i, j = divmod(pid, num_patches)
            top_left = (j * patch_size, i * patch_size)
            bottom_right = ((j + 1) * patch_size, (i + 1) * patch_size)
            draw.rectangle([top_left, bottom_right], fill=color + (int(255 * alpha),))

    return Image.alpha_composite(image, overlay).convert("RGB")

def visualize_significant_patches_mask(image, patch_ids, patch_size=14, alpha=0.5, color=(255, 255, 255)):
    """
    Highlights specified patches with semi-transparent overlay.
    """
    overlay_group = [(patch_ids, color)]
    return draw_patches_overlay(image, overlay_group, patch_size, alpha)

def task_relevant_selection(multihead_attention, image, significant_patches, top_k=120, delete_ratio=0.0, return_delete=False):
    """
    Highlights and compares significant patches with top attention patches.

    Returns reusable static visual-token positions. When ``return_delete`` is True,
    also returns the lowest-attention subset of reusable candidates as delete positions.
    Positions are LLM visual-token positions (patch id + 1 for OpenVLA).
    """
    attn_score = token_attention_merge(multihead_attention)
    top_patches = get_top_attention_patches(attn_score, top_k)
    attn_values = attn_score.cpu().numpy() if isinstance(attn_score, torch.Tensor) else attn_score

    grid_size = 224 // 14
    all_ids = set(range(grid_size * grid_size))
    static_set = set(significant_patches or [])
    task_set = set(top_patches)
    dynamic_set = all_ids - static_set

    # dynamic/task visualization: recompute = dynamic ∪ task
    dynamic_only = dynamic_set - task_set
    task_only = task_set - dynamic_set
    overlap = dynamic_set & task_set

    # Three-way visualization: dynamic-only, task-only, overlap
    patch_groups = [
        (dynamic_only, (35, 166, 213)),  # dynamic-only (cyan)
        (task_only, (155, 89, 182)),     # task-only (purple)
        (overlap, (231, 111, 81)),       # overlap (orange)
    ]

    result_image = draw_patches_overlay(image, patch_groups, patch_size=14, alpha=0.4)

    reusable_candidates = sorted(static_set - task_set)
    delete_count = int(len(reusable_candidates) * max(0.0, min(float(delete_ratio), 1.0)))
    if delete_count > 0:
        delete_patch_ids = sorted(
            reusable_candidates,
            key=lambda pid: float(attn_values[pid]) if 0 <= pid < len(attn_values) else 0.0,
        )[:delete_count]
        delete_set = set(delete_patch_ids)
    else:
        delete_set = set()

    reuse_patch_ids = sorted(pid for pid in reusable_candidates if pid not in delete_set)
    v_token_start = 1
    remaining = sorted([pid + v_token_start for pid in reuse_patch_ids])
    # Keep deletion ordered by increasing attention so layer-wise ratios delete the least relevant patches first.
    deleted = [pid + v_token_start for pid in delete_patch_ids] if delete_count > 0 else []

    if return_delete:
        return np.array(result_image), remaining, deleted
    return np.array(result_image), remaining
