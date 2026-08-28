#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


COSMOS_SERVER_URL = "http://127.0.0.1:8001/predict"
COSMOS_PROMPT = "You are an autonomous vehicle planning system."
DOMAIN_NAME = "av"
IMAGE_SIZE = 480
ACTION_CHUNK_SIZE = 30
FPS = 5.0


def encode_image_base64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode input image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def cosmos_predict_latent(frame_bgr: np.ndarray, action: np.ndarray, latent_output_path: str | Path, output_mode: str = "latent") -> dict[str, Any]:
    if action.shape != (ACTION_CHUNK_SIZE, 9):
        raise ValueError(f"Expected action shape ({ACTION_CHUNK_SIZE}, 9), got {action.shape}")
    if output_mode not in {"latent", "video", "both"}:
        raise ValueError(f"Unsupported output_mode={output_mode!r}")

    latent_output_path = str(latent_output_path)
    payload = {
        "image": encode_image_base64(frame_bgr),
        "prompt": COSMOS_PROMPT,
        "domain_name": DOMAIN_NAME,
        "image_size": IMAGE_SIZE,
        "action": action.tolist(),
        "output_mode": output_mode,
    }
    if output_mode in {"latent", "both"}:
        payload["latent_output_path"] = latent_output_path

    print("sending Cosmos request to:", COSMOS_SERVER_URL)
    print("output_mode:", output_mode)
    if output_mode in {"latent", "both"}:
        print("latent_output_path:", latent_output_path)

    response = requests.post(COSMOS_SERVER_URL, json=payload, timeout=(10, 1800))
    print("Cosmos HTTP status:", response.status_code)
    if response.status_code != 200:
        print(response.text)
        raise RuntimeError("Cosmos request failed.")

    result = response.json()
    if "error" in result:
        raise RuntimeError(result["error"])

    if output_mode in {"latent", "both"}:
        if result.get("latent_path") is None:
            raise RuntimeError(f"Cosmos response missing latent_path: {result}")
        print("saved latent:", result["latent_path"])
        print("latent shape:", result.get("latent_shape"))
        print("latent dtype:", result.get("latent_dtype"))

    if output_mode in {"video", "both"}:
        print("returned frames:", len(result.get("video", [])))
        print("returned fps:", float(result.get("fps", FPS)))

    return result


def load_action_json(action_path: str | Path) -> np.ndarray:
    with open(action_path, "r", encoding="utf-8") as f:
        action = np.asarray(json.load(f), dtype=np.float32)
    if action.shape != (ACTION_CHUNK_SIZE, 9):
        raise ValueError(f"Expected action shape ({ACTION_CHUNK_SIZE}, 9), got {action.shape}")
    return action


def read_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame

def read_image_frame(image_root: str | Path, frame_idx: int) -> np.ndarray:
    image_root = Path(image_root)

    image_paths = sorted(
        p for p in image_root.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not image_paths:
        raise RuntimeError(f"No images found in: {image_root}")

    if frame_idx < 0 or frame_idx >= len(image_paths):
        raise IndexError(f"frame_idx={frame_idx} out of range, num_images={len(image_paths)}")

    image_path = image_paths[frame_idx]
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    print(f"read frame {frame_idx}: {image_path}")

    return frame


if __name__ == "__main__":
    IMAGE_ROOT = "lerobot_data_r2r_50/videos/chunk-000/observation.images.rgb.100cm_0deg"
    FRAME_IDX = 0

    ACTION_PATH = "datasets/rxr_sub/trajectory_ranking_xy/ranking_actions/ep000000_s0_f000000/gt.json"
    LATENT_PATH = "datasets/rxr_sub/trajectory_ranking_xy/tmp_latents/ep000000_s0_f000000_gt.pt"

    frame = read_image_frame(IMAGE_ROOT, FRAME_IDX)
    action = load_action_json(ACTION_PATH)

    result = cosmos_predict_latent(
        frame,
        action,
        LATENT_PATH,
        output_mode="latent",
    )

    print("done")
    print(result)