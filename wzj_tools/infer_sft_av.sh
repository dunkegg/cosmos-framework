RUN_DIR=wzj_train_output/cosmos3_action_fd/action_sft/action_fd_r2r_posttrain
CKPT=$RUN_DIR/checkpoints/$(cat $RUN_DIR/checkpoints/latest_checkpoint.txt)
torchrun --nproc-per-node=4 -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i inputs/omni/action_forward_dynamics_batch_real_case_turn.jsonl \
  -o outputs/sft_fd_av_r2r \
  --checkpoint-path "$CKPT" \
  --config-file "$RUN_DIR/config.yaml" \
  --seed=0