
python -m wzj_tools.build_trajectory_ranking_xy_data_v3 \
  --annotations datasets/rxr_sub/subtasks.jsonl \
  --dataset-root /mnt/ws_nas/data_5880/lerobot_data_r2r_50 \
  --video-key observation.images.rgb.100cm_0deg \
  --nav-key local_traj \
  --action-chunk-size 30 \
  --starts-per-subtask 3 \
  --min-future-frames 0 \
  --output-root datasets/rxr_sub/trajectory_ranking_xy_chunksize_30

# python -m wzj_tools.generate_all_ranking_latents \
#   --manifest datasets/rxr_sub/trajectory_ranking_xy/trajectory_ranking_manifest.jsonl \
#   --chunk-size 128 \
#   --latent-dtype float16 \
#   --max-groups 0