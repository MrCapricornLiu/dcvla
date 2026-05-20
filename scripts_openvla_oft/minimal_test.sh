#!/usr/bin/env bash
set -e

ts=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/home/lch/Documents/dcvla/results_oft"
logdir="${BASE_PATH}/${ts}_oft_minimal"
mkdir -p "$logdir"

echo "🚀 OpenVLA-OFT minimal test -> Log: $logdir"

run_one() {
  local task=$1
  local ckpt=$2
  local logfile="$logdir/${task}.log"
  echo "  ↳ 任务 ${task} -> Log: ${logfile}"
  CUDA_VISIBLE_DEVICES=0 python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py     --pretrained_checkpoint "$ckpt"     --task_suite_name "$task"     --local_log_dir "$logdir"     --num_trials_per_task 1     --num_tasks 1     --task_start_id 0     --vit_cache_standalone True     --use_vla_cache True     --use_vit_cache True     --vit_cache_reuse True     --vit_cache_patch_metric cosine     --vit_cache_sim_threshold 0.992     --vit_cache_attention_top_k 80     --vit_cache_static_top_k 120     --vit_cache_keyframe_interval 5     > "$logfile" 2>&1
}

run_one libero_spatial checkpoints/openvla-7b-oft-finetuned-libero-spatial
run_one libero_object checkpoints/openvla-7b-oft-finetuned-libero-object
run_one libero_goal checkpoints/openvla-7b-oft-finetuned-libero-goal
run_one libero_10 checkpoints/openvla-7b-oft-finetuned-libero-10

echo "✅ OpenVLA-OFT minimal test completed: $logdir"
