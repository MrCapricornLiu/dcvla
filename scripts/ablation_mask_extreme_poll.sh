#!/usr/bin/env bash
# GPU-polling script for extreme mask inconsistency ablation.
# 8 runs total: OpenVLA (4) + OpenVLA-OFT (4).

set -e

ts=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/home/lch/Documents/dcvla/results_ablation_extreme"

CHECK_INTERVAL=10
COOLDOWN=40
WAIT_LOG_INTERVAL=40
MONITOR_GPUS=(0 1 3 4)

get_deploy_id() {
  local mon_id=$1
  if [ "$mon_id" -eq 3 ]; then
    echo 2
  elif [ "$mon_id" -eq 4 ]; then
    echo 3
  else
    echo "$mon_id"
  fi
}

get_threshold() {
  local mon_id=$1
  echo 20000
}

find_free_gpu() {
  for id in "${MONITOR_GPUS[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $id 2>/dev/null)
    if [ -z "$used" ]; then
      used=999999
    fi
    thr=$(get_threshold "$id")
    if [ "$used" -lt "$thr" ]; then
      echo "$id"
      return 0
    fi
  done
  return 1
}

run_task() {
  local cmd="$1"
  local found=""
  local last_log=0
  while true; do
    found=$(find_free_gpu || true)
    if [ -n "$found" ]; then
      break
    fi
    now=$(date +%s)
    if [ $((now - last_log)) -ge $WAIT_LOG_INTERVAL ]; then
      echo "⏳ 等待空闲GPU... (阈值=2000MB)"
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true
      last_log=$now
    fi
    sleep $CHECK_INTERVAL
  done
  local deploy_id
  deploy_id=$(get_deploy_id "$found")
  echo "✅ 使用GPU 监听ID: $found -> 部署ID: $deploy_id"
  CUDA_VISIBLE_DEVICES=$deploy_id bash -c "$cmd" &
  sleep $COOLDOWN
}

run_suite_openvla() {
  local task=$1
  local ckpt=$2
  local logdir=$3
  shift 3
  local extra_flags="$*"

  local logfile="$logdir/${task}.log"
  echo "  ↳ 任务 ${task} -> Log: ${logfile}"

  run_task "source /home/lch/miniconda3/etc/profile.d/conda.sh && conda activate dcvla && cd /home/lch/Documents/dcvla/src/openvla && python /home/lch/Documents/dcvla/src/openvla/experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${ckpt} \
    --task_suite_name ${task} \
    --local_log_dir ${logdir} \
    --num_trials_per_task 20 \
    --num_tasks 10 \
    --task_start_id 0 \
    --vit_cache_standalone True \
    --use_vla_cache True \
    --use_vit_cache True \
    --vit_cache_reuse True \
    ${extra_flags} \
    > ${logfile} 2>&1"
}

run_suite_oft() {
  local task=$1
  local ckpt=$2
  local logdir=$3
  shift 3
  local extra_flags="$*"

  local logfile="$logdir/${task}.log"
  echo "  ↳ 任务 ${task} -> Log: ${logfile}"

  run_task "source /home/lch/miniconda3/etc/profile.d/conda.sh && conda activate dcvlaoft && cd /home/lch/Documents/dcvla/src/openvla-oft && python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${ckpt} \
    --task_suite_name ${task} \
    --local_log_dir ${logdir} \
    --num_trials_per_task 20 \
    --num_tasks 10 \
    --task_start_id 0 \
    --vit_cache_standalone True \
    --use_vla_cache True \
    --use_vit_cache True \
    --vit_cache_reuse True \
    ${extra_flags} \
    > ${logfile} 2>&1"
}

run_group_openvla() {
  local tag=$1
  local extra_flags="$2"
  local logdir="${BASE_PATH}/${ts}_${tag}"
  mkdir -p "$logdir"
  echo "🚀 启动参数组 ${tag} -> Log: $logdir"
  run_suite_openvla libero_spatial /home/lch/Documents/dcvla/src/openvla/checkpoints/openvla-7b-finetuned-libero-spatial "$logdir" "$extra_flags"
  run_suite_openvla libero_object  /home/lch/Documents/dcvla/src/openvla/checkpoints/openvla-7b-finetuned-libero-object  "$logdir" "$extra_flags"
  run_suite_openvla libero_goal    /home/lch/Documents/dcvla/src/openvla/checkpoints/openvla-7b-finetuned-libero-goal    "$logdir" "$extra_flags"
  run_suite_openvla libero_10      /home/lch/Documents/dcvla/src/openvla/checkpoints/openvla-7b-finetuned-libero-10      "$logdir" "$extra_flags"
}

run_group_oft() {
  local tag=$1
  local extra_flags="$2"
  local logdir="${BASE_PATH}/${ts}_${tag}"
  mkdir -p "$logdir"
  echo "🚀 启动参数组 ${tag} -> Log: $logdir"
  run_suite_oft libero_spatial /home/lch/Documents/dcvla/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial "$logdir" "$extra_flags"
  run_suite_oft libero_object  /home/lch/Documents/dcvla/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-object  "$logdir" "$extra_flags"
  run_suite_oft libero_goal    /home/lch/Documents/dcvla/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-goal    "$logdir" "$extra_flags"
  run_suite_oft libero_10      /home/lch/Documents/dcvla/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-10      "$logdir" "$extra_flags"
}

# -------------------------
# Extreme mask inconsistency settings
# -------------------------
# OpenVLA (cosine)
OPENVLA_BASE="--vit_cache_patch_metric cosine --vit_cache_sim_threshold 0.992 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 160 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric cosine --llm_cache_sim_threshold 0.992 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 160"
OPENVLA_VIT_LOW="--vit_cache_patch_metric cosine --vit_cache_sim_threshold 0.988 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 160 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric cosine --llm_cache_sim_threshold 0.992 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 160"
OPENVLA_LLM_LOW="--vit_cache_patch_metric cosine --vit_cache_sim_threshold 0.992 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 160 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric cosine --llm_cache_sim_threshold 0.988 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 160"
OPENVLA_VIT_HIGH="--vit_cache_patch_metric cosine --vit_cache_sim_threshold 0.997 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 160 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric cosine --llm_cache_sim_threshold 0.992 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 160"
OPENVLA_LLM_HIGH="--vit_cache_patch_metric cosine --vit_cache_sim_threshold 0.992 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 160 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric cosine --llm_cache_sim_threshold 0.997 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 160"

# OpenVLA-OFT (gray diff)
OFT_BASE="--vit_cache_patch_metric gray_diff --vit_cache_gray_diff_threshold 0.004 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 120 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric gray_diff --llm_cache_gray_diff_threshold 0.004 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 120"
OFT_VIT_LOW="--vit_cache_patch_metric gray_diff --vit_cache_gray_diff_threshold 0.003 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 120 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric gray_diff --llm_cache_gray_diff_threshold 0.004 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 120"
OFT_LLM_LOW="--vit_cache_patch_metric gray_diff --vit_cache_gray_diff_threshold 0.004 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 120 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric gray_diff --llm_cache_gray_diff_threshold 0.003 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 120"
OFT_VIT_HIGH="--vit_cache_patch_metric gray_diff --vit_cache_gray_diff_threshold 0.007 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 120 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric gray_diff --llm_cache_gray_diff_threshold 0.004 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 120"
OFT_LLM_HIGH="--vit_cache_patch_metric gray_diff --vit_cache_gray_diff_threshold 0.004 --vit_cache_attention_top_k 160 --vit_cache_static_top_k 120 --vit_cache_keyframe_interval 5 \
  --llm_cache_patch_metric gray_diff --llm_cache_gray_diff_threshold 0.007 --llm_cache_attention_top_k 160 --llm_cache_static_top_k 120"

# -------------------------
# Run OpenVLA-OFT first
# -------------------------
# run_group_oft "oft_mask_vit_low"  "$OFT_VIT_LOW"
# run_group_oft "oft_mask_llm_low"  "$OFT_LLM_LOW"
# run_group_oft "oft_mask_vit_high" "$OFT_VIT_HIGH"
# run_group_oft "oft_mask_llm_high" "$OFT_LLM_HIGH"

# -------------------------
# Run OpenVLA
# -------------------------
run_group_openvla "ovla_mask_vit_low"  "$OPENVLA_VIT_LOW"
run_group_openvla "ovla_mask_llm_low"  "$OPENVLA_LLM_LOW"
run_group_openvla "ovla_mask_vit_high" "$OPENVLA_VIT_HIGH"
run_group_openvla "ovla_mask_llm_high" "$OPENVLA_LLM_HIGH"

echo "✅ All 8 runs dispatched. Check ${BASE_PATH}/${ts}_*/ for logs."
