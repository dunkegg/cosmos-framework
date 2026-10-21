# Reward V2 — Qwen3-VL instruction-conditioned temporal reward

Copy four Python files to `wzj_tools/reward_model/` without replacing existing V1 files.

## Model path
Use the actual Hugging Face snapshot directory containing `config.json`:
`.../models--Qwen--Qwen3-VL-8B-Instruct/snapshots/<hash>`.

## 1. Encode instructions (Qwen required only here)
```bash
python -m wzj_tools.reward_model.encode_instructions \
  --manifest datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/trajectory_ranking_manifest.jsonl \
  --qwen-path /ABS/PATH/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/REVISION \
  --output datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/instruction_embeddings_qwen3vl.pt \
  --text-field both
```

## 2. Train (no Qwen weights needed; V1 warm start)
```bash
python -m wzj_tools.reward_model.train_v2 \
  --manifest datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/trajectory_ranking_manifest.jsonl \
  --embeddings datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/instruction_embeddings_qwen3vl.pt \
  --init-v1 wzj_train_output/reward_v1_temporal_8782/best.pt \
  --output-dir wzj_train_output/reward_v2_qwen \
  --batch-size 8 --epochs 30 --lr 1e-4 --split-seed 42
```

## 3. Evaluate
```bash
python -m wzj_tools.reward_model.eval_v2 \
  --manifest datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/trajectory_ranking_manifest.jsonl \
  --embeddings datasets/rxr_sub/trajectory_ranking_xy_chunksize_60/instruction_embeddings_qwen3vl.pt \
  --checkpoint wzj_train_output/reward_v2_qwen/best.pt
```

## Caveats
- Pairwise GT-vs-perturbed supervision does not force the model to use language. Run instruction-swap and action-goal conflict tests before claiming language grounding.
- The old V1 checkpoint's episode split is only reproduced if the manifest and dataset code are unchanged. Compare printed validation episode IDs.
- Qwen3-VL support requires a recent Transformers version. Verify `from transformers import Qwen3VLForConditionalGeneration` before running.
- V1 initialization reproduces the original visual function at initialization because FiLM is zero-initialized.
