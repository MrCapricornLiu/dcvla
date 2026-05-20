#!/usr/bin/env bash
ts=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/home/lch/Documents/dcvla/results_oft"

CHECK_INTERVAL=10
COOLDOWN=30
WAIT_LOG_INTERVAL=30
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
      echo "⏳ 等待空闲GPU... (阈值=20000MB)"
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

logdir="${BASE_PATH}/${ts}_g01_cos_0992_k80_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g01_cos_0992_k80_s120_kf5 | metric=cosine, sim=0.992, gray=-, rgb=-, attn_topk=80, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g02_cos_0993_k100_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g02_cos_0993_k100_s120_kf5 | metric=cosine, sim=0.993, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g03_cos_0994_k120_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g03_cos_0994_k120_s120_kf5 | metric=cosine, sim=0.994, gray=-, rgb=-, attn_topk=120, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g04_cos_0995_k140_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g04_cos_0995_k140_s120_kf5 | metric=cosine, sim=0.995, gray=-, rgb=-, attn_topk=140, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python /home/lch/Documents/dcvla/src/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 10 \
  --num_tasks 2 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"
