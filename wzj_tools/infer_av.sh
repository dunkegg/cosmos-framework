export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    -i inputs/omni/action_forward_dynamics_av_local.json \
    -o outputs/fd_av \
    --checkpoint-path /mnt/ws_nas/wzj/cosmos-framework/checkpoints/Cosmos3-Nano \
    --seed=0