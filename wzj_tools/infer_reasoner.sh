torchrun --nproc-per-node=4 -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    -i inputs/reasoner/nav_reward_test.json \
    -o outputs/reasoner_nav \
    --checkpoint-path /mnt/nas/wzj/cosmos-framework/checkpoints/Cosmos3-Nano \
    --seed=0