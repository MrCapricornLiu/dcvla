#!/usr/bin/env bash
ts=$(date +"%Y%m%d_%H%M%S")
BASE_PATH="/home/lch/Documents/dcvla/results"

CHECK_INTERVAL=10
COOLDOWN=80
WAIT_LOG_INTERVAL=60
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

# logdir="${BASE_PATH}/${ts}_g01_cos_0992_k80_s120_kf5"
# mkdir -p "$logdir"
# export logdir
# echo "🚀 启动参数组 g01_cos_0992_k80_s120_kf5 | metric=cosine, sim=0.992, gray=-, rgb=-, attn_topk=80, static_topk=120, kf=5 -> Log: $logdir"
# echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
# run_task "python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --local_log_dir $logdir \
#   --num_trials_per_task 2 \
#   --num_tasks 2 \
#   --task_start_id 0 \
#   --vit_cache_standalone True \
#   --use_vla_cache True \
#   --use_vit_cache True \
#   --vit_cache_reuse True \
#   --vit_cache_patch_metric cosine \
#   --vit_cache_sim_threshold 0.992 \
#   --vit_cache_attention_top_k 80 \
#   --vit_cache_static_top_k 120 \
#   --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

# echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
# run_task "python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
#   --task_suite_name libero_object \
#   --local_log_dir $logdir \
#   --num_trials_per_task 2 \
#   --num_tasks 2 \
#   --task_start_id 0 \
#   --vit_cache_standalone True \
#   --use_vla_cache True \
#   --use_vit_cache True \
#   --vit_cache_reuse True \
#   --vit_cache_patch_metric cosine \
#   --vit_cache_sim_threshold 0.992 \
#   --vit_cache_attention_top_k 80 \
#   --vit_cache_static_top_k 120 \
#   --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

# echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
# run_task "python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
#   --task_suite_name libero_goal \
#   --local_log_dir $logdir \
#   --num_trials_per_task 2 \
#   --num_tasks 2 \
#   --task_start_id 0 \
#   --vit_cache_standalone True \
#   --use_vla_cache True \
#   --use_vit_cache True \
#   --vit_cache_reuse True \
#   --vit_cache_patch_metric cosine \
#   --vit_cache_sim_threshold 0.992 \
#   --vit_cache_attention_top_k 80 \
#   --vit_cache_static_top_k 120 \
#   --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

# echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
# run_task "python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
#   --task_suite_name libero_10 \
#   --local_log_dir $logdir \
#   --num_trials_per_task 2 \
#   --num_tasks 2 \
#   --task_start_id 0 \
#   --vit_cache_standalone True \
#   --use_vla_cache True \
#   --use_vit_cache True \
#   --vit_cache_reuse True \
#   --vit_cache_patch_metric cosine \
#   --vit_cache_sim_threshold 0.992 \
#   --vit_cache_attention_top_k 80 \
#   --vit_cache_static_top_k 120 \
#   --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

# logdir="${BASE_PATH}/${ts}_g02_cos_0993_k100_s120_kf5"
# mkdir -p "$logdir"
# export logdir
# echo "🚀 启动参数组 g02_cos_0993_k100_s120_kf5 | metric=cosine, sim=0.993, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
# echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
# run_task "python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --local_log_dir $logdir \
#   --num_trials_per_task 2 \
#   --num_tasks 2 \
#   --task_start_id 0 \
#   --vit_cache_standalone True \
#   --use_vla_cache True \
#   --use_vit_cache True \
#   --vit_cache_reuse True \
#   --vit_cache_patch_metric cosine \
#   --vit_cache_sim_threshold 0.993 \
#   --vit_cache_attention_top_k 100 \
#   --vit_cache_static_top_k 120 \
#   --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

# echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
# run_task "python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
#   --task_suite_name libero_object \
#   --local_log_dir $logdir \
#   --num_trials_per_task 2 \
#   --num_tasks 2 \
#   --task_start_id 0 \
#   --vit_cache_standalone True \
#   --use_vla_cache True \
#   --use_vit_cache True \
#   --vit_cache_reuse True \
#   --vit_cache_patch_metric cosine \
#   --vit_cache_sim_threshold 0.993 \
#   --vit_cache_attention_top_k 100 \
#   --vit_cache_static_top_k 120 \
#   --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

# echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
# run_task "python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
#   --task_suite_name libero_goal \
#   --local_log_dir $logdir \
#   --num_trials_per_task 2 \
#   --num_tasks 2 \
#   --task_start_id 0 \
#   --vit_cache_standalone True \
#   --use_vla_cache True \
#   --use_vit_cache True \
#   --vit_cache_reuse True \
#   --vit_cache_patch_metric cosine \
#   --vit_cache_sim_threshold 0.993 \
#   --vit_cache_attention_top_k 100 \
#   --vit_cache_static_top_k 120 \
#   --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

# echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
# run_task "python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
#   --task_suite_name libero_10 \
#   --local_log_dir $logdir \
#   --num_trials_per_task 2 \
#   --num_tasks 2 \
#   --task_start_id 0 \
#   --vit_cache_standalone True \
#   --use_vla_cache True \
#   --use_vit_cache True \
#   --vit_cache_reuse True \
#   --vit_cache_patch_metric cosine \
#   --vit_cache_sim_threshold 0.993 \
#   --vit_cache_attention_top_k 100 \
#   --vit_cache_static_top_k 120 \
#   --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g03_cos_0994_k120_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g03_cos_0994_k120_s120_kf5 | metric=cosine, sim=0.994, gray=-, rgb=-, attn_topk=120, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
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
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
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
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
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
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
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
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
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
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
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
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
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
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
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

logdir="${BASE_PATH}/${ts}_g05_cos_0996_k160_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g05_cos_0996_k160_s120_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=160, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g06_cos_0997_k80_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g06_cos_0997_k80_s120_kf5 | metric=cosine, sim=0.997, gray=-, rgb=-, attn_topk=80, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g07_cos_0998_k100_s120_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g07_cos_0998_k100_s120_kf3 | metric=cosine, sim=0.998, gray=-, rgb=-, attn_topk=100, static_topk=120, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g08_cos_0992_k120_s120_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g08_cos_0992_k120_s120_kf3 | metric=cosine, sim=0.992, gray=-, rgb=-, attn_topk=120, static_topk=120, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g09_cos_0993_k140_s120_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g09_cos_0993_k140_s120_kf8 | metric=cosine, sim=0.993, gray=-, rgb=-, attn_topk=140, static_topk=120, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g10_cos_0994_k160_s120_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g10_cos_0994_k160_s120_kf8 | metric=cosine, sim=0.994, gray=-, rgb=-, attn_topk=160, static_topk=120, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g11_cos_0995_k80_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g11_cos_0995_k80_s160_kf5 | metric=cosine, sim=0.995, gray=-, rgb=-, attn_topk=80, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g12_cos_0996_k100_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g12_cos_0996_k100_s160_kf5 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=100, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g13_cos_0997_k120_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g13_cos_0997_k120_s160_kf5 | metric=cosine, sim=0.997, gray=-, rgb=-, attn_topk=120, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g14_cos_0998_k140_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g14_cos_0998_k140_s160_kf5 | metric=cosine, sim=0.998, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.998 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g15_cos_0992_k160_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g15_cos_0992_k160_s160_kf5 | metric=cosine, sim=0.992, gray=-, rgb=-, attn_topk=160, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.992 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g16_cos_0993_k80_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g16_cos_0993_k80_s160_kf5 | metric=cosine, sim=0.993, gray=-, rgb=-, attn_topk=80, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.993 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g17_cos_0994_k100_s160_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g17_cos_0994_k100_s160_kf3 | metric=cosine, sim=0.994, gray=-, rgb=-, attn_topk=100, static_topk=160, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.994 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g18_cos_0995_k120_s160_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g18_cos_0995_k120_s160_kf3 | metric=cosine, sim=0.995, gray=-, rgb=-, attn_topk=120, static_topk=160, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.995 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g19_cos_0996_k140_s160_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g19_cos_0996_k140_s160_kf8 | metric=cosine, sim=0.996, gray=-, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.996 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g20_cos_0997_k160_s160_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g20_cos_0997_k160_s160_kf8 | metric=cosine, sim=0.997, gray=-, rgb=-, attn_topk=160, static_topk=160, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric cosine \
  --vit_cache_sim_threshold 0.997 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g21_gray_002_k80_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g21_gray_002_k80_s120_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=80, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g22_gray_0025_k100_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g22_gray_0025_k100_s120_kf5 | metric=gray_diff, sim=-, gray=0.025, rgb=-, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g23_gray_003_k120_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g23_gray_003_k120_s120_kf5 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=120, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g24_gray_0035_k140_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g24_gray_0035_k140_s120_kf5 | metric=gray_diff, sim=-, gray=0.035, rgb=-, attn_topk=140, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g25_gray_004_k160_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g25_gray_004_k160_s120_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=160, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g26_gray_005_k80_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g26_gray_005_k80_s120_kf5 | metric=gray_diff, sim=-, gray=0.05, rgb=-, attn_topk=80, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g27_gray_006_k100_s120_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g27_gray_006_k100_s120_kf3 | metric=gray_diff, sim=-, gray=0.06, rgb=-, attn_topk=100, static_topk=120, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.06 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.06 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.06 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.06 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g28_gray_002_k120_s120_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g28_gray_002_k120_s120_kf3 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=120, static_topk=120, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g29_gray_0025_k140_s120_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g29_gray_0025_k140_s120_kf8 | metric=gray_diff, sim=-, gray=0.025, rgb=-, attn_topk=140, static_topk=120, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g30_gray_003_k160_s120_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g30_gray_003_k160_s120_kf8 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=160, static_topk=120, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g31_gray_0035_k80_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g31_gray_0035_k80_s160_kf5 | metric=gray_diff, sim=-, gray=0.035, rgb=-, attn_topk=80, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g32_gray_004_k100_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g32_gray_004_k100_s160_kf5 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=100, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g33_gray_005_k120_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g33_gray_005_k120_s160_kf5 | metric=gray_diff, sim=-, gray=0.05, rgb=-, attn_topk=120, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g34_gray_006_k140_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g34_gray_006_k140_s160_kf5 | metric=gray_diff, sim=-, gray=0.06, rgb=-, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.06 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.06 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.06 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.06 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g35_gray_002_k160_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g35_gray_002_k160_s160_kf5 | metric=gray_diff, sim=-, gray=0.02, rgb=-, attn_topk=160, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.02 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g36_gray_0025_k80_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g36_gray_0025_k80_s160_kf5 | metric=gray_diff, sim=-, gray=0.025, rgb=-, attn_topk=80, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.025 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g37_gray_003_k100_s160_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g37_gray_003_k100_s160_kf3 | metric=gray_diff, sim=-, gray=0.03, rgb=-, attn_topk=100, static_topk=160, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.03 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g38_gray_0035_k120_s160_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g38_gray_0035_k120_s160_kf3 | metric=gray_diff, sim=-, gray=0.035, rgb=-, attn_topk=120, static_topk=160, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.035 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g39_gray_004_k140_s160_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g39_gray_004_k140_s160_kf8 | metric=gray_diff, sim=-, gray=0.04, rgb=-, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.04 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g40_gray_005_k160_s160_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g40_gray_005_k160_s160_kf8 | metric=gray_diff, sim=-, gray=0.05, rgb=-, attn_topk=160, static_topk=160, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric gray_diff \
  --vit_cache_gray_diff_threshold 0.05 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g41_rgb_004_k80_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g41_rgb_004_k80_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=80, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g42_rgb_005_k100_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g42_rgb_005_k100_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=100, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g43_rgb_006_k120_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g43_rgb_006_k120_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=120, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g44_rgb_007_k140_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g44_rgb_007_k140_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.07, attn_topk=140, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g45_rgb_008_k160_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g45_rgb_008_k160_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.08, attn_topk=160, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g46_rgb_004_k80_s120_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g46_rgb_004_k80_s120_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=80, static_topk=120, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g47_rgb_005_k100_s120_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g47_rgb_005_k100_s120_kf3 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=100, static_topk=120, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g48_rgb_006_k120_s120_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g48_rgb_006_k120_s120_kf3 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=120, static_topk=120, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g49_rgb_007_k140_s120_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g49_rgb_007_k140_s120_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.07, attn_topk=140, static_topk=120, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g50_rgb_008_k160_s120_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g50_rgb_008_k160_s120_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.08, attn_topk=160, static_topk=120, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 120 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g51_rgb_004_k80_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g51_rgb_004_k80_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=80, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g52_rgb_005_k100_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g52_rgb_005_k100_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=100, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g53_rgb_006_k120_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g53_rgb_006_k120_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=120, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g54_rgb_007_k140_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g54_rgb_007_k140_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.07, attn_topk=140, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g55_rgb_008_k160_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g55_rgb_008_k160_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.08, attn_topk=160, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g56_rgb_004_k80_s160_kf5"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g56_rgb_004_k80_s160_kf5 | metric=rgb_diff, sim=-, gray=-, rgb=0.04, attn_topk=80, static_topk=160, kf=5 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.04 \
  --vit_cache_attention_top_k 80 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 5 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g57_rgb_005_k100_s160_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g57_rgb_005_k100_s160_kf3 | metric=rgb_diff, sim=-, gray=-, rgb=0.05, attn_topk=100, static_topk=160, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.05 \
  --vit_cache_attention_top_k 100 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g58_rgb_006_k120_s160_kf3"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g58_rgb_006_k120_s160_kf3 | metric=rgb_diff, sim=-, gray=-, rgb=0.06, attn_topk=120, static_topk=160, kf=3 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.06 \
  --vit_cache_attention_top_k 120 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 3 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g59_rgb_007_k140_s160_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g59_rgb_007_k140_s160_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.07, attn_topk=140, static_topk=160, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.07 \
  --vit_cache_attention_top_k 140 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"

logdir="${BASE_PATH}/${ts}_g60_rgb_008_k160_s160_kf8"
mkdir -p "$logdir"
export logdir
echo "🚀 启动参数组 g60_rgb_008_k160_s160_kf8 | metric=rgb_diff, sim=-, gray=-, rgb=0.08, attn_topk=160, static_topk=160, kf=8 -> Log: $logdir"
echo "  ↳ 任务 libero_spatial -> Log: $logdir/libero_spatial.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_spatial.log 2>&1"

echo "  ↳ 任务 libero_object -> Log: $logdir/libero_object.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_object.log 2>&1"

echo "  ↳ 任务 libero_goal -> Log: $logdir/libero_goal.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_goal.log 2>&1"

echo "  ↳ 任务 libero_10 -> Log: $logdir/libero_10.log"
run_task "python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir $logdir \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True \
  --vit_cache_patch_metric rgb_diff \
  --vit_cache_rgb_diff_threshold 0.08 \
  --vit_cache_attention_top_k 160 \
  --vit_cache_static_top_k 160 \
  --vit_cache_keyframe_interval 8 > $logdir/libero_10.log 2>&1"
