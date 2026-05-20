# 获取当前执行时间（格式：YYYYmmdd_HHMMSS）
ts=$(date +"%Y%m%d_%H%M%S")

# 原始基路径
base="/home/lch/Documents/dcvla/results"

# 生成带时间戳的新路径
logdir="${base}/${ts}"

# 创建目录
mkdir -p "$logdir"



logfile="${logdir}/ff.log"
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 5 \
  --num_tasks 10 \
  --task_start_id 0 \
  --use_vla_cache False \
  --use_vit_cache False > "$logfile" 2>&1 &



logfile="${logdir}/ft.log"
VLA_VIT_DEBUG=0 CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 5 \
  --num_tasks 10 \
  --task_start_id 0 \
  --vit_cache_standalone True \
  --use_vla_cache False \
  --use_vit_cache True > "$logfile" 2>&1 &



logfile="${logdir}/tf.log"
CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 5 \
  --num_tasks 10 \
  --task_start_id 0 \
  --use_vla_cache True \
  --use_vit_cache False > "$logfile" 2>&1 &



logfile="${logdir}/tt.log"
CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --local_log_dir "$logdir" \
  --num_trials_per_task 5 \
  --num_tasks 10 \
  --task_start_id 0 \
  --use_vla_cache True \
  --use_vit_cache True > "$logfile" 2>&1 &