from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/bridge-v2-subset-synthetic-captions",
    repo_type="dataset",
    local_dir="examples/data/bridge-v2-subset-synthetic-captions",
    max_workers=1,
)