"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from datetime import datetime

import filecmp
import shutil
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub import HfApi, hf_hub_download
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from .vla_cache_utils import find_static_patches, task_relevant_selection, get_layer_mask_schedule

from transformers import DynamicCache

# Debug flag for optional verbose logging
DEBUG_VIT = os.environ.get("VLA_VIT_DEBUG", "0") == "1"

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
LATENCY_SEGMENTS = ("overhead", "vit", "llm", "action_head")

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _latency_start():
    use_cuda = torch.cuda.is_available()
    start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
    if use_cuda:
        torch.cuda.synchronize()
        start_event.record()
    return time.perf_counter(), start_event, use_cuda


def _latency_stop(timer_state):
    wall_start, start_event, use_cuda = timer_state
    if use_cuda:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = start_event.elapsed_time(end_event)
    else:
        cuda_ms = 0.0
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return wall_ms, cuda_ms


def _finalize_latency_metrics(overhead_ms, overhead_cuda_ms, model_ms, model_cuda_ms, model_metrics):
    metrics = {}
    model_metrics = model_metrics or {}
    for segment in LATENCY_SEGMENTS:
        metrics[f"{segment}_wall_ms"] = float(model_metrics.get(f"{segment}_wall_ms", 0.0))
        metrics[f"{segment}_cuda_ms"] = float(model_metrics.get(f"{segment}_cuda_ms", 0.0))

    model_inner_wall_ms = sum(metrics[f"{segment}_wall_ms"] for segment in ("vit", "llm", "action_head"))
    model_inner_cuda_ms = sum(metrics[f"{segment}_cuda_ms"] for segment in ("vit", "llm", "action_head"))
    metrics["overhead_wall_ms"] = float(overhead_ms) + max(0.0, float(model_ms) - model_inner_wall_ms)
    metrics["overhead_cuda_ms"] = float(overhead_cuda_ms) + max(0.0, float(model_cuda_ms) - model_inner_cuda_ms)
    metrics["total_wall_ms"] = sum(metrics[f"{segment}_wall_ms"] for segment in LATENCY_SEGMENTS)
    metrics["total_cuda_ms"] = sum(metrics[f"{segment}_cuda_ms"] for segment in LATENCY_SEGMENTS)
    metrics["model_wall_ms"] = float(model_ms)
    metrics["model_cuda_ms"] = float(model_cuda_ms)
    return metrics


def _format_latency_metrics(metrics):
    return (
        "[Latency] wall_ms total={total_wall_ms:.3f} overhead={overhead_wall_ms:.3f} "
        "vit={vit_wall_ms:.3f} llm={llm_wall_ms:.3f} action_head={action_head_wall_ms:.3f} | "
        "cuda_ms total={total_cuda_ms:.3f} overhead={overhead_cuda_ms:.3f} "
        "vit={vit_cuda_ms:.3f} llm={llm_cuda_ms:.3f} action_head={action_head_cuda_ms:.3f}"
    ).format(**metrics)


def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False
    

def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')

def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of
    modeling_prismatic.py, configuration_prismatic.py, and local helper modules:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {
        "modeling_prismatic.py": None,
        "configuration_prismatic.py": None,
        "llm_bucket_graph.py": None,
    }

    openvla_root = Path(__file__).resolve().parents[2]
    for root, _, files in os.walk(openvla_root / "prismatic"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


    
def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)


    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def process_image(image, crop_scale=0.9, batch_size=1):
    
    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")

    return image

def get_vla_action(cfg, vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, last_caches=None):
    """Generates an action with the VLA policy."""
    overhead_timer = _latency_start()
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")
    
    result_image = image
    prev_image = Image.fromarray(obs["prev_image"])
    prompt_cache = last_caches['past_key_values'] if last_caches is not None else None
    vit_cache = last_caches.get('vit_cache') if last_caches is not None else None
    prev_attn = last_caches['attentions'] if last_caches is not None else None
    mask_indices_vit = None
    mask_indices_delete_vit = None
    mask_indices_llm = None
    mask_indices_delete_llm = None
    vla.language_model.config.proportion_attn_var = None
    vla.language_model.config.reusable_patches = None
    vla.language_model.config.deleted_patches = None
    vla.language_model.config.current_deleted_patches = None
    cache_overhead_benchmark = bool(getattr(cfg, "cache_overhead_benchmark", True))
    vit_overhead_benchmark = cache_overhead_benchmark or bool(getattr(cfg, "vit_cache_benchmark", False))
    llm_overhead_benchmark = cache_overhead_benchmark or bool(getattr(cfg, "llm_cache_benchmark", False))
    vla.language_model.config.vla_cache_effective = bool(cfg.use_vla_cache)
    vla.language_model.config.vla_delete_effective = bool(getattr(cfg, "llm_delete_enable", False))
    vla.language_model.config.vla_overhead_benchmark = llm_overhead_benchmark
    vla.config.llm_bucket_graph_enable = bool(getattr(cfg, "llm_bucket_graph_enable", False))
    vla.config.llm_bucket_graph_capture = bool(getattr(cfg, "llm_bucket_graph_capture", True))
    vla.config.llm_bucket_graph_warmup = int(getattr(cfg, "llm_bucket_graph_warmup", 2))
    vla.config.llm_bucket_graph_max_graphs = int(getattr(cfg, "llm_bucket_graph_max_graphs", 32))
    vla.config.llm_bucket_graph_fallback = bool(getattr(cfg, "llm_bucket_graph_fallback", True))


    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        image = process_image(image)
        prev_image = process_image(prev_image)

    reuse_mask_full = None
    # Always run ViT mask setup (benchmark or real cache)
    if True:
        if cfg.use_vit_cache:
            mode_msg = ">> ViT cache + VLA on" if cfg.use_vla_cache else ">> ViT cache + VLA off"
        else:
            mode_msg = ">> ViT benchmark + VLA on" if cfg.use_vla_cache else ">> ViT benchmark + VLA off"
        print(mode_msg)

        stable_patches_vit = None
        stable_patches_llm = None
        if prompt_cache is not None or cfg.vit_cache_standalone or cfg.vit_cache_benchmark or cfg.use_vla_cache:
            def _resolve_llm_param(name: str, default):
                val = getattr(cfg, f"llm_cache_{name}", None)
                return default if val is None else val

            vit_metric = getattr(cfg, "vit_cache_patch_metric", "cosine")
            llm_metric = _resolve_llm_param("patch_metric", vit_metric)

            vit_sim = getattr(cfg, "vit_cache_sim_threshold", 0.996)
            vit_diff = getattr(cfg, "vit_cache_patch_diff_threshold", 0.1)
            vit_gray = getattr(cfg, "vit_cache_gray_diff_threshold", vit_diff)
            vit_rgb = getattr(cfg, "vit_cache_rgb_diff_threshold", vit_diff)

            llm_sim = _resolve_llm_param("sim_threshold", vit_sim)
            llm_diff = _resolve_llm_param("patch_diff_threshold", vit_diff)
            llm_gray = _resolve_llm_param("gray_diff_threshold", vit_gray)
            llm_rgb = _resolve_llm_param("rgb_diff_threshold", vit_rgb)

            vit_static_top_k = getattr(cfg, "vit_cache_static_top_k", 130)
            llm_static_top_k = _resolve_llm_param("static_top_k", vit_static_top_k)

            def _get_thresholds(metric, sim_thr, gray_thr, rgb_thr, diff_thr):
                if metric == "gray_diff":
                    return None, gray_thr if gray_thr is not None else diff_thr
                if metric == "rgb_diff":
                    return None, rgb_thr if rgb_thr is not None else diff_thr
                return sim_thr, diff_thr

            vit_sim_thr, vit_diff_thr = _get_thresholds(vit_metric, vit_sim, vit_gray, vit_rgb, vit_diff)
            llm_sim_thr, llm_diff_thr = _get_thresholds(llm_metric, llm_sim, llm_gray, llm_rgb, llm_diff)

            stable_patches_vit = find_static_patches(
                image,
                prev_image,
                top_k=vit_static_top_k,
                metric=vit_metric,
                sim_threshold=vit_sim_thr if vit_sim_thr is not None else 0.996,
                diff_threshold=vit_diff_thr,
            )
            stable_patches_llm = find_static_patches(
                image,
                prev_image,
                top_k=llm_static_top_k,
                metric=llm_metric,
                sim_threshold=llm_sim_thr if llm_sim_thr is not None else 0.996,
                diff_threshold=llm_diff_thr,
            )

        if prev_attn is not None:
            vit_attn_top_k = getattr(cfg, "vit_cache_attention_top_k", 120)
            llm_attn_top_k = _resolve_llm_param("attention_top_k", vit_attn_top_k)
            vit_delete_ratio = (
                getattr(cfg, "vit_delete_ratio", 0.0)
                if (getattr(cfg, "vit_delete_enable", False) or vit_overhead_benchmark)
                else 0.0
            )
            llm_delete_ratio = (
                getattr(cfg, "llm_delete_ratio", 0.0)
                if (getattr(cfg, "llm_delete_enable", False) or llm_overhead_benchmark)
                else 0.0
            )

            result_image, remaining_static_tokens_vit, delete_tokens_vit = task_relevant_selection(
                prev_attn,
                image,
                stable_patches_vit,
                top_k=vit_attn_top_k,
                delete_ratio=vit_delete_ratio,
                return_delete=True,
            )
            _, remaining_static_tokens_llm, delete_tokens_llm = task_relevant_selection(
                prev_attn,
                image,
                stable_patches_llm,
                top_k=llm_attn_top_k,
                delete_ratio=llm_delete_ratio,
                return_delete=True,
            )
            skip_tokens_vit = remaining_static_tokens_vit + delete_tokens_vit
            mask_indices_vit = (
                torch.tensor(skip_tokens_vit, device=DEVICE)
                if skip_tokens_vit
                else None
            )
            mask_indices_delete_vit = (
                torch.tensor(delete_tokens_vit, device=DEVICE)
                if delete_tokens_vit
                else None
            )
            mask_indices_llm = (
                torch.tensor(remaining_static_tokens_llm, device=DEVICE)
                if remaining_static_tokens_llm
                else None
            )
            mask_indices_delete_llm = (
                torch.tensor(delete_tokens_llm, device=DEVICE)
                if delete_tokens_llm
                else None
            )

            if cfg.use_vla_cache or llm_overhead_benchmark:
                vla.language_model.config.reusable_patches = mask_indices_llm
                vla.language_model.config.deleted_patches = mask_indices_delete_llm
                vla.language_model.config.proportion_attn_var = get_layer_mask_schedule(prev_attn)

        if not cfg.use_vla_cache and not llm_overhead_benchmark:
            # honor flag: do not reuse LLaMA cache when VLA-Cache is off and no overhead-control path is requested
            prompt_cache = None

    else:
        print(">> VLA-Cache disabled")
        prompt_cache = None
        mask_indices_llm = None

    # Configure ViT KV cache reuse (static patch K/V)
    vit_cache_out = None
    enable_vit = True  # always run ViT cache pipeline (benchmark or real reuse)
    if enable_vit:
        # Helpers to set cache/mask on each featurizer
        def _prepare_featurizer(featurizer, cache_payload, mask_idx, delete_idx):
            num_prefix = getattr(featurizer, "num_prefix_tokens", 0)
            num_patches = featurizer.patch_embed.num_patches
            reuse_mask_local = torch.zeros(num_prefix + num_patches, dtype=torch.bool, device=DEVICE)
            delete_mask_local = torch.zeros_like(reuse_mask_local)
            if mask_idx is not None:
                # LLM visual positions are patch_id + 1; map back to ViT patch ids.
                valid_idx = mask_idx - 1
                valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < num_patches)]
                reuse_mask_local[num_prefix + valid_idx] = True
            if delete_idx is not None:
                valid_delete_idx = delete_idx - 1
                valid_delete_idx = valid_delete_idx[(valid_delete_idx >= 0) & (valid_delete_idx < num_patches)]
                delete_mask_local[num_prefix + valid_delete_idx] = True
                reuse_mask_local[num_prefix + valid_delete_idx] = True
            vit_cache_path_enabled = bool(cfg.use_vit_cache or vit_overhead_benchmark)
            vit_static_reuse_effective = bool(cfg.use_vit_cache and getattr(cfg, "vit_cache_reuse", True))
            vit_delete_effective = bool(cfg.use_vit_cache and getattr(cfg, "vit_delete_enable", False))
            # In benchmark mode we keep the internal featurizer cache within an episode but do not serialize it in last_caches.
            if last_caches is None and vit_cache_path_enabled and hasattr(featurizer, "reset_vla_cache"):
                featurizer.reset_vla_cache()
            use_cache_payload = (cache_payload is not None and cfg.use_vit_cache and not cfg.vit_cache_benchmark)
            if use_cache_payload:
                cache_state = cache_payload
            elif vit_cache_path_enabled and last_caches is not None:
                cache_state = featurizer.get_vla_cache_state()
            else:
                cache_state = [None] * len(featurizer.blocks)
            featurizer.set_vla_cache_state(
                cache_state,
                reuse_mask_local,
                enable_reuse=vit_cache_path_enabled,
                enable_static_reuse=vit_static_reuse_effective,
                keyframe_interval=getattr(cfg, "vit_cache_keyframe_interval", 0),
                delete_mask=delete_mask_local if (getattr(cfg, "vit_delete_enable", False) or vit_overhead_benchmark) else None,
                enable_delete=vit_delete_effective,
                overhead_benchmark=vit_overhead_benchmark,
            )
            static_count = reuse_mask_local.sum().item()
            delete_count = delete_mask_local.sum().item()
            total_count = reuse_mask_local.numel()
            ratio = static_count / max(1, total_count)
            delete_ratio = delete_count / max(1, total_count)
            print(f"[ViT Reuse] skip={static_count}/{total_count} ({ratio:.3f}) prune={delete_count}/{total_count} ({delete_ratio:.3f})")
            if DEBUG_VIT:
                print(f"[ViT-Setup] mask_len={reuse_mask_local.numel()} skip={reuse_mask_local.sum().item()} prune={delete_mask_local.sum().item()} num_prefix={num_prefix} num_patches={num_patches}")
            return reuse_mask_local

        # Primary featurizer
        reuse_mask_full = _prepare_featurizer(
            vla.vision_backbone.featurizer,
            None if vit_cache is None else vit_cache.get("alpha"),
            mask_indices_vit,
            mask_indices_delete_vit,
        )

        # Fused backbone (if exists) uses同样的缓存/掩码策略
        if getattr(vla.vision_backbone, "use_fused_vision_backbone", False):
            _prepare_featurizer(
                vla.vision_backbone.fused_featurizer,
                None if vit_cache is None else vit_cache.get("beta"),
                mask_indices_vit,
                mask_indices_delete_vit,
            )
    else:
        # Ensure stale cache not reused
        if hasattr(vla.vision_backbone.featurizer, "reset_vla_cache"):
            vla.vision_backbone.featurizer.reset_vla_cache()
        if getattr(vla.vision_backbone, "use_fused_vision_backbone", False):
            if hasattr(vla.vision_backbone.fused_featurizer, "reset_vla_cache"):
                vla.vision_backbone.fused_featurizer.reset_vla_cache()

    if prompt_cache is None:
        prompt_cache = DynamicCache()
        
    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    overhead_wall_ms, overhead_cuda_ms = _latency_stop(overhead_timer)
    model_timer = _latency_start()
    action, last_caches = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False, return_dict_in_generate=True, 
                                                        output_attentions = True, past_key_values=prompt_cache)
    model_wall_ms, model_cuda_ms = _latency_stop(model_timer)
    post_overhead_timer = _latency_start()
    # Collect ViT cache for next frame
    if cfg.use_vit_cache and not cfg.vit_cache_benchmark:
        vit_cache_out = {
            "alpha": vla.vision_backbone.featurizer.get_vla_cache_state()
        }
        if getattr(vla.vision_backbone, "use_fused_vision_backbone", False):
            vit_cache_out["beta"] = vla.vision_backbone.fused_featurizer.get_vla_cache_state()
        last_caches["vit_cache"] = vit_cache_out
   
    result_image = np.array(result_image)
    post_overhead_wall_ms, post_overhead_cuda_ms = _latency_stop(post_overhead_timer)
    latency_metrics = _finalize_latency_metrics(
        overhead_wall_ms + post_overhead_wall_ms,
        overhead_cuda_ms + post_overhead_cuda_ms,
        model_wall_ms,
        model_cuda_ms,
        getattr(vla, "_vla_latency_metrics", None),
    )
    print(_format_latency_metrics(latency_metrics))
    return action, last_caches, result_image, latency_metrics
