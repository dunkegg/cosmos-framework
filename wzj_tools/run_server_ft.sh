export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun \
    --nproc-per-node=4 \
    -m cosmos_framework.scripts.forward_dynamics_server_av \
    --parallelism-preset=throughput \
    --checkpoint-path /mnt/nas/wzj/cosmos-framework/wzj_train_output/cosmos3_action_fd/action_sft/action_fd_r2r_posttrain/checkpoints/iter_000020000 \
    --port 8001 \
    --domain-name av \
    --fps 10 \
    --action-chunk-size 60 \
    --raw-action-dim 9 \
    --image-size 480 \
    --num-steps 30 \
    --seed 0