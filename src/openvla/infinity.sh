#!/bin/bash

# ================= 配置区域 =================
# 显存空闲阈值 (MB)，低于此值视为“空闲”
THRESHOLD=1500
# 每次检测失败后的重试等待时间 (秒)
CHECK_INTERVAL=10
# 任务启动后的冷却时间 (秒) —— 关键！必须等待 Python 真正占用了显存才能进行下一次检测
COOLDOWN=60

# 基础路径
BASE_PATH="/home/lch/Documents/dcvla/results/"

# 这里的 ID 是用来【监听】的 (nvidia-smi -i X)
# 对应你的服务器物理槽位 0, 1, 3, 4
MONITOR_GPUS=(0 1 3 4)

# ===========================================

# 显卡 ID 映射函数
# 输入: 监听 ID (0/1/3/4)
# 输出: 部署 ID (0/1/2/3)
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

# 查找空闲 GPU 的函数
find_free_gpu() {
    for id in "${MONITOR_GPUS[@]}"; do
        # 查询显存占用
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $id)

        # 如果占用小于阈值，说明这就有一张空卡，直接返回它的 ID
        if [ "$used" -lt "$THRESHOLD" ]; then
            echo "$id"
            return 0
        fi
    done
    return 1
}

echo "========================================================"
echo "   动态排队系统启动 (9任务版)"
echo "   监听 GPU: ${MONITOR_GPUS[*]}"
echo "   映射逻辑: 3->2, 4->3"
echo "========================================================"

# 定义总共 9 个任务，使用循环逐个处理
for TASK_INDEX in {1..9}; do

    echo ""
    echo ">>> [任务进度 $TASK_INDEX/9] 正在寻找可用的 GPU..."

    # --- 1. 死循环：直到找到一张空卡 ---
    FOUND_MON_ID=""
    while true; do
        # 尝试获取一个空闲的监听 ID
        FOUND_MON_ID=$(find_free_gpu)

        if [ -n "$FOUND_MON_ID" ]; then
            # 找到了！
            break
        else
            sleep $CHECK_INTERVAL
        fi
    done

    # --- 2. ID 映射与准备 ---
    # 将监听 ID (如 3) 转换为 部署 ID (如 2)
    DEPLOY_ID=$(get_deploy_id "$FOUND_MON_ID")

    echo -e "\n✅ 捕获到空闲显卡！监听ID: $FOUND_MON_ID -> 部署ID: $DEPLOY_ID"

    # 生成时间戳
    ts=$(date +"%Y%m%d_%H%M%S")

    # --- 3. 根据任务序号执行对应的命令 ---
    case $TASK_INDEX in
        # ==========================================
        # 第一组：Interval 3 系列 (原有的5个任务)
        # ==========================================
        1)
            # Spatial, Interval 3, Gray, 0.03
            logdir="${BASE_PATH}/${ts}_spatial_vitcache-gray0.03-keyframe3_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 1: Spatial (Interval 3, Gray) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
              --task_suite_name libero_spatial \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 3 \
              --vit_cache_patch_metric "gray_diff" \
              --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

        2)
            # Long (Libero 10), Interval 3, RGB, 0.05
            logdir="${BASE_PATH}/${ts}_long_vitcache-rgb0.05-keyframe3_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 2: Long (Interval 3, RGB) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
              --task_suite_name libero_10 \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 3 \
              --vit_cache_patch_metric "rgb_diff" \
              --vit_cache_rgb_diff_threshold 0.05 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

        3)
            # Goal, Interval 3, RGB, 0.05
            logdir="${BASE_PATH}/${ts}_goal_vitcache-rgb0.05-keyframe3_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 3: Goal (Interval 3, RGB) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
              --task_suite_name libero_goal \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 3 \
              --vit_cache_patch_metric "rgb_diff" \
              --vit_cache_rgb_diff_threshold 0.05 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

        4)
            # Object, Interval 3, RGB, 0.05
            logdir="${BASE_PATH}/${ts}_object_vitcache-rgb0.05-keyframe3_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 4: Object (Interval 3, RGB) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
              --task_suite_name libero_object \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 3 \
              --vit_cache_patch_metric "rgb_diff" \
              --vit_cache_rgb_diff_threshold 0.05 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

        5)
            # Spatial, Interval 3, RGB, 0.05
            logdir="${BASE_PATH}/${ts}_spatial_vitcache-rgb0.05-keyframe3_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 5: Spatial (Interval 3, RGB) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
              --task_suite_name libero_spatial \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 3 \
              --vit_cache_patch_metric "rgb_diff" \
              --vit_cache_rgb_diff_threshold 0.05 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

        # ==========================================
        # 第二组：Interval 5 系列 (新增的4个任务)
        # ==========================================
        6)
            # Long (Libero 10), Interval 5, Gray, 0.03
            logdir="${BASE_PATH}/${ts}_long_vitcache-gray0.03-keyframe5_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 6: Long (Interval 5, Gray) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
              --task_suite_name libero_10 \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 5 \
              --vit_cache_patch_metric "gray_diff" \
              --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

        7)
            # Goal, Interval 5, Gray, 0.03
            logdir="${BASE_PATH}/${ts}_goal_vitcache-gray0.03-keyframe5_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 7: Goal (Interval 5, Gray) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
              --task_suite_name libero_goal \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 5 \
              --vit_cache_patch_metric "gray_diff" \
              --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

        8)
            # Object, Interval 5, Gray, 0.03
            logdir="${BASE_PATH}/${ts}_object_vitcache-gray0.03-keyframe5_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 8: Object (Interval 5, Gray) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object \
              --task_suite_name libero_object \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 5 \
              --vit_cache_patch_metric "gray_diff" \
              --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

        9)
            # Spatial, Interval 5, Gray, 0.03
            logdir="${BASE_PATH}/${ts}_spatial_vitcache-gray0.03-keyframe5_llmcache-default"
            mkdir -p "$logdir"
            echo "🚀 启动任务 9: Spatial (Interval 5, Gray) -> Log: $logdir"

            CUDA_VISIBLE_DEVICES=$DEPLOY_ID python experiments/robot/libero/run_libero_eval.py \
              --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
              --task_suite_name libero_spatial \
              --local_log_dir "$logdir" \
              --num_trials_per_task 20 \
              --num_tasks 10 \
              --task_start_id 0 \
              --vit_cache_standalone True \
              --use_vla_cache True \
              --use_vit_cache True \
              --vit_cache_reuse True \
              --vit_cache_keyframe_interval 5 \
              --vit_cache_patch_metric "gray_diff" \
              --vit_cache_gray_diff_threshold 0.03 > "${logdir}/vlaTrue_vitTrue.log" 2>&1 &
            ;;

    esac

    # --- 4. 关键冷却时间 ---
    echo "⏳ 任务已提交后台。脚本休眠 ${COOLDOWN} 秒，等待 GPU 显存占用上升..."
    sleep $COOLDOWN
    echo "--------------------------------------------------------"

done

echo "🎉 所有 9 个任务均已派发完毕！脚本即将退出 (后台任务仍在运行)。"
wait