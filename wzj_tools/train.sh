export DATASET_PATH=lerobot_data_r2r_50
# export BASE_CHECKPOINT_PATH=/mnt/nas/wzj/cosmos-framework/checkpoints/Cosmos3-Nano-DCP
export BASE_CHECKPOINT_PATH=/mnt/nas/wzj/cosmos-framework/wzj_train_output/cosmos3_action_fd/action_sft/action_fd_r2r_posttrain_9_8/checkpoints/iter_000040000
export WAN_VAE_PATH=/mnt/nas/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/mnt/nas/wzj/cosmos-framework/wzj_train_output
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export TORCHDYNAMO_DISABLE=1

torchrun \
  --nproc_per_node=4 \
  --master_port=29501 \
  -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_fd_av_r2r_posttrain.toml