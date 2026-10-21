export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun \
    --nproc-per-node=4 \
    -m cosmos_framework.scripts.forward_dynamics_server_av \
    --parallelism-preset=throughput \
    --checkpoint-path /mnt/ws_nas/wzj/cosmos-framework/checkpoints/Cosmos3-Nano \
    --port 8001 \
    --domain-name av \
    --fps 5 \
    --action-chunk-size 30 \
    --raw-action-dim 9 \
    --image-size 480 \
    --num-steps 30 \
    --seed 0