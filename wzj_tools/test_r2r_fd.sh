export DATASET_PATH=lerobot_data_r2r_50
export BASE_CHECKPOINT_PATH=/mnt/nas/wzj/cosmos-framework/checkpoints/Cosmos3-Nano-DCP
export WAN_VAE_PATH=/mnt/nas/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth

RUN_DIR=wzj_train_output/cosmos3_action_fd/action_sft/action_fd_r2r_posttrain
CKPT=$RUN_DIR/checkpoints/$(cat $RUN_DIR/checkpoints/latest_checkpoint.txt)

python -m wzj_tools.test_r2r_fd \
  --dataset-root lerobot_data_r2r_50 \
  --checkpoint "$CKPT" \
  --start-episode 0 \
  --num-episodes 50 \
  --start-frame 0 \
  --num-gpus 4 \
  --master-port 29511 \
  --output-dir outputs/r2r_fd_test_40000_lr \
  --flip-lr