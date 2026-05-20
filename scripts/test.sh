# 获取当前执行时间（格式：YYYYmmdd_HHMMSS）
ts=$(date +"%Y%m%d_%H%M%S")

# 原始基路径
base="/home/lch/Documents/dcvla/results/"

# 生成带时间戳的新路径
logdir="${base}/${ts}_forfigure"

# 创建目录
mkdir -p "$logdir"



logfile="${logdir}/1.log"
CUDA_VISIBLE_DEVICES=0 nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 1 \
  --num_tasks 1 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache False \
  --use_vit_cache True \
  --vit_cache_reuse False > "$logfile" 2>&1 &

logfile="${logdir}/2.log"
CUDA_VISIBLE_DEVICES=1 nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --local_log_dir "$logdir" \
  --num_trials_per_task 1 \
  --num_tasks 1 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache False \
  --use_vit_cache True \
  --vit_cache_reuse True > "$logfile" 2>&1 &


logfile="${logdir}/3.log"
CUDA_VISIBLE_DEVICES=2 nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --local_log_dir "$logdir" \
  --num_trials_per_task 1 \
  --num_tasks 1 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse False > "$logfile" 2>&1 &


logfile="${logdir}/4.log"
CUDA_VISIBLE_DEVICES=3 nohup python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --local_log_dir "$logdir" \
  --num_trials_per_task 1 \
  --num_tasks 1 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache True \
  --use_vit_cache True \
  --vit_cache_reuse True > "$logfile" 2>&1 &
