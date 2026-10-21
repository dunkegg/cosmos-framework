
# python -m wzj_tools.build_trajectory_ranking \
#   --annotations datasets/rxr_sub/subtasks.jsonl \
#   --dataset-root lerobot_data_r2r_50 \
#   --video-key observation.images.rgb.100cm_0deg \
#   --nav-key local_traj \
#   --action-chunk-size 60 \
#   --starts-per-subtask 3 \
#   --min-future-frames 10 \
#   --output-root datasets/rxr_sub/trajectory_ranking_xy_chunksize_60

# python -m wzj_tools.extract_ranking_latents \
#   --manifest datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/trajectory_ranking_manifest.jsonl \
#   --chunk-size 128 \
#   --latent-dtype bfloat16 \
#   --max-groups 0

python -m wzj_tools.reward_model.encode_instructions \
  --manifest datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/trajectory_ranking_manifest.jsonl \
  --qwen-path /mnt/nas/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --output datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/instruction_embeddings_qwen3vl.pt \
  --text-field both