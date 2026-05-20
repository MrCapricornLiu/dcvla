CKPT=/home/lch/Documents/dcvla/src/SimplerEnv-OpenVLA/checkpoints/openvla-7b
LOGDIR=/home/lch/Documents/dcvla/results_simplerenv/$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOGDIR"

# baseline: Move Near
CUDA_VISIBLE_DEVICES=0 python /home/lch/Documents/dcvla/src/SimplerEnv-OpenVLA/simpler_env/main_inference.py \
  --policy-model openvla \
  --ckpt-path "$CKPT" \
  --policy-setup google_robot \
  --env-name MoveNearGoogleBakedTexInScene-v1 \
  --scene-name google_pick_coke_can_1_v4 \
  --use-vla-cache False \
  --use-vit-cache False \
  > "$LOGDIR/move_near_baseline.log" 2>&1 &

# CTR: Move Near
CUDA_VISIBLE_DEVICES=1 python /home/lch/Documents/dcvla/src/SimplerEnv-OpenVLA/simpler_env/main_inference.py \
  --policy-model openvla \
  --ckpt-path "$CKPT" \
  --policy-setup google_robot \
  --env-name MoveNearGoogleBakedTexInScene-v1 \
  --scene-name google_pick_coke_can_1_v4 \
  --use-vla-cache True \
  --use-vit-cache True \
  --vit-cache-reuse True \
  --vit-cache-keyframe-interval 5 \
  --vit-cache-patch-metric cosine \
  --vit-cache-sim-threshold 0.992 \
  --vit-cache-attention-top-k 160 \
  --vit-cache-static-top-k 160 \
  > "$LOGDIR/move_near_ctr.log" 2>&1 &

# baseline: Drawer
CUDA_VISIBLE_DEVICES=2 python /home/lch/Documents/dcvla/src/SimplerEnv-OpenVLA/simpler_env/main_inference.py \
  --policy-model openvla \
  --ckpt-path "$CKPT" \
  --policy-setup google_robot \
  --env-name OpenDrawerCustomInScene-v0 \
  --scene-name dummy_drawer \
  --use-vla-cache False \
  --use-vit-cache False \
  > "$LOGDIR/drawer_baseline.log" 2>&1 &

# CTR: Drawer
CUDA_VISIBLE_DEVICES=3 python /home/lch/Documents/dcvla/src/SimplerEnv-OpenVLA/simpler_env/main_inference.py \
  --policy-model openvla \
  --ckpt-path "$CKPT" \
  --policy-setup google_robot \
  --env-name OpenDrawerCustomInScene-v0 \
  --scene-name dummy_drawer \
  --use-vla-cache True \
  --use-vit-cache True \
  --vit-cache-reuse True \
  --vit-cache-keyframe-interval 5 \
  --vit-cache-patch-metric cosine \
  --vit-cache-sim-threshold 0.992 \
  --vit-cache-attention-top-k 160 \
  --vit-cache-static-top-k 160 \
  > "$LOGDIR/drawer_ctr.log" 2>&1 &
