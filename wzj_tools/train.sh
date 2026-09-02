export DATASET_PATH=lerobot_data_r2r_50
export BASE_CHECKPOINT_PATH=/mnt/ws_nas/wzj/cosmos-framework/checkpoints/Cosmos3-Nano-DCP
export WAN_VAE_PATH=/mnt/ws_nas/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth

PYTHONPATH=. python -m cosmos_framework.scripts.train \
    --sft-toml examples/toml/sft_config/action_fd_av_r2r_posttrain.toml \
    --dryrun