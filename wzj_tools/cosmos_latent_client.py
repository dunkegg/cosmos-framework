#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

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
ACTION_CHUNK_SIZE = 60
ACTION_DIM = 9
FPS = 10.0

def save_base64_video(
    video_b64: list[str],
    output_path: str | Path,
    fps: float,
) -> None:
    if not video_b64:
        raise RuntimeError("Empty video returned by Cosmos server.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for frame_b64 in video_b64:
        raw = base64.b64decode(frame_b64)
        frame = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            raise RuntimeError("Failed to decode returned PNG frame.")
        frames.append(frame)

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video: {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()

    print(
        f"saved video: {output_path} "
        f"frames={len(frames)} fps={fps} size={w}x{h}"
    )
def encode_image_base64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode input image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def get_server_info() -> dict[str, Any]:
    info_url = COSMOS_SERVER_URL.rsplit("/", 1)[0] + "/info"
    response = requests.get(info_url, timeout=10)
    response.raise_for_status()
    return response.json()


def check_server_config() -> dict[str, Any]:
    info = get_server_info()
    if int(info["action_chunk_size"]) != ACTION_CHUNK_SIZE:
        raise RuntimeError(f"action_chunk_size mismatch: client={ACTION_CHUNK_SIZE}, server={info['action_chunk_size']}")
    if int(info["raw_action_dim"]) != ACTION_DIM:
        raise RuntimeError(f"raw_action_dim mismatch: client={ACTION_DIM}, server={info['raw_action_dim']}")
    if float(info["fps"]) != FPS:
        raise RuntimeError(f"fps mismatch: client={FPS}, server={info['fps']}")
    if str(info["domain"]) != DOMAIN_NAME:
        raise RuntimeError(f"domain mismatch: client={DOMAIN_NAME}, server={info['domain']}")
    print(f"Cosmos server config OK: fps={info['fps']} chunk={info['action_chunk_size']} dim={info['raw_action_dim']} domain={info['domain']}")
    return info

def cosmos_predict_latent(
    frame_bgr: np.ndarray,
    action: np.ndarray,
    latent_output_path: str | Path | None = None,
    video_output_path: str | Path | None = None,
    output_mode: str = "latent",
) -> dict[str, Any]:
    if action.shape != (ACTION_CHUNK_SIZE, ACTION_DIM):
        raise ValueError(
            f"Expected action shape ({ACTION_CHUNK_SIZE}, {ACTION_DIM}), "
            f"got {action.shape}"
        )

    if output_mode not in {"latent", "video", "both"}:
        raise ValueError(f"Unsupported output_mode={output_mode!r}")

    payload = {
        "image": encode_image_base64(frame_bgr),
        "prompt": COSMOS_PROMPT,
        "domain_name": DOMAIN_NAME,
        "image_size": IMAGE_SIZE,
        "action": action.tolist(),
        "output_mode": output_mode,
    }

    # --------------------------------------------------------
    # Latent output
    # Server directly saves latent to this path.
    # --------------------------------------------------------
    if output_mode in {"latent", "both"}:
        if latent_output_path is None:
            raise ValueError(
                "latent_output_path is required "
                "for output_mode='latent' or 'both'"
            )

        latent_output_path = Path(latent_output_path)
        latent_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload["latent_output_path"] = str(
            latent_output_path
        )

    print("sending Cosmos request to:", COSMOS_SERVER_URL)
    print("output_mode:", output_mode)

    if output_mode in {"latent", "both"}:
        print(
            "latent_output_path:",
            latent_output_path,
        )

    if (
        output_mode in {"video", "both"}
        and video_output_path is not None
    ):
        print(
            "video_output_path:",
            video_output_path,
        )

    # --------------------------------------------------------
    # Cosmos inference
    # --------------------------------------------------------
    response = requests.post(
        COSMOS_SERVER_URL,
        json=payload,
        timeout=(10, 1800),
    )

    print(
        "Cosmos HTTP status:",
        response.status_code,
    )

    if response.status_code != 200:
        print(response.text)
        raise RuntimeError(
            "Cosmos request failed."
        )

    result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])

    # --------------------------------------------------------
    # Latent result
    # --------------------------------------------------------
    if output_mode in {"latent", "both"}:
        latent_path = result.get(
            "latent_path"
        )

        if latent_path is None:
            raise RuntimeError(
                f"Cosmos response missing latent_path: {result}"
            )

        print(
            "saved latent:",
            latent_path,
        )
        print(
            "latent shape:",
            result.get("latent_shape"),
        )
        print(
            "latent dtype:",
            result.get("latent_dtype"),
        )
        print(
            "seed:",
            result.get("seed"),
        )

    # --------------------------------------------------------
    # Video result
    # Server returns:
    #
    #   video = [
    #       "<base64 PNG>",
    #       "<base64 PNG>",
    #       ...
    #   ]
    #
    # --------------------------------------------------------
    if output_mode in {"video", "both"}:
        video_b64 = result.get(
            "video",
            [],
        )

        fps = float(
            result.get("fps", FPS)
        )

        print(
            "returned frames:",
            len(video_b64),
        )
        print(
            "returned fps:",
            fps,
        )

        # ----------------------------------------------------
        # Save returned PNG frames as MP4
        # ----------------------------------------------------
        if video_output_path is not None:
            if not video_b64:
                raise RuntimeError(
                    "Cosmos returned empty video."
                )

            video_output_path = Path(
                video_output_path
            )
            video_output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            frames = []

            for frame_b64 in video_b64:
                raw = base64.b64decode(
                    frame_b64
                )

                frame = cv2.imdecode(
                    np.frombuffer(
                        raw,
                        dtype=np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                )

                if frame is None:
                    raise RuntimeError(
                        "Failed to decode "
                        "returned PNG frame."
                    )

                frames.append(frame)

            h, w = frames[0].shape[:2]

            writer = cv2.VideoWriter(
                str(video_output_path),
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                fps,
                (w, h),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Failed to create video: "
                    f"{video_output_path}"
                )

            for frame in frames:
                if frame.shape[:2] != (h, w):
                    raise RuntimeError(
                        "Returned video frames "
                        "have inconsistent sizes."
                    )

                writer.write(frame)

            writer.release()

            print(
                f"saved video: {video_output_path} "
                f"frames={len(frames)} "
                f"fps={fps} "
                f"size={w}x{h}"
            )

    return result


def load_action_json(action_path: str | Path, expected_chunk_size: int = ACTION_CHUNK_SIZE, expected_action_dim: int = ACTION_DIM) -> np.ndarray:
    with open(action_path, "r", encoding="utf-8") as f:
        action = np.asarray(json.load(f), dtype=np.float32)
    if action.shape != (expected_chunk_size, expected_action_dim):
        raise ValueError(f"Expected action shape ({expected_chunk_size}, {expected_action_dim}), got {action.shape}")
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
    image_paths = sorted(p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
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
    check_server_config()
    frame = read_image_frame(IMAGE_ROOT, FRAME_IDX)
    action = load_action_json(ACTION_PATH)
    result = cosmos_predict_latent(frame, action, LATENT_PATH, output_mode="latent")
    print("done")
    print(result)
