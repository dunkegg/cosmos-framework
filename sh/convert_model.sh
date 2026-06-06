BASE_CHECKPOINT_NAME=Cosmos3-Nano   # or Cosmos3-Super — match the recipe in Step 1

# Default output dir matches the launcher (see Step 3 → Option A to override).
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/$BASE_CHECKPOINT_NAME \
  --checkpoint-path $BASE_CHECKPOINT_NAME