# 获取当前执行时间（格式：YYYYmmdd_HHMMSS）
ts=$(date +"%Y%m%d_%H%M%S")

# 原始基路径
base="/home/lch/Documents/dcvla/results_oft/"

# 生成带时间戳的新路径
logdir="${base}/${ts}_oft_baselineof4tasks"

# 创建目录
mkdir -p "$logdir"


logfile="${logdir}/spatial_baseline.log"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache False \
  --use_vit_cache True \
  --vit_cache_reuse False > "$logfile" 2>&1 &

logfile="${logdir}/goal_baseline.log"
CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache False \
  --use_vit_cache True \
  --vit_cache_reuse False > "$logfile" 2>&1 &

logfile="${logdir}/object_baseline.log"
CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache False \
  --use_vit_cache True \
  --vit_cache_reuse False > "$logfile" 2>&1 &

logfile="${logdir}/10_baseline.log"
CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 20 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache False \
  --use_vit_cache True \
  --vit_cache_reuse False > "$logfile" 2>&1 &