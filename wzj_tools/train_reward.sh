python -m wzj_tools.reward_model.train_v2 \
  --manifest datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/trajectory_ranking_manifest.jsonl \
  --embeddings datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/instruction_embeddings_qwen3vl.pt \
  --init-v1 wzj_train_output/reward_v1_temporal_8782/best.pt \
  --output-dir wzj_train_output/reward_v2_qwen \
  --batch-size 8 \
  --epochs 30 \
  --lr 1e-4 \
  --split-seed 42