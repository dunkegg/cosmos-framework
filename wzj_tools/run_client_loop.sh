#!/usr/bin/env bash

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] start client"
    python test_forward_dynamics_server_av.py &
    python test_qwen_vllm_client.py &
    sleep 60
done