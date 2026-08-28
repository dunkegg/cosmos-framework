import base64
import json
from pathlib import Path

import cv2
import numpy as np
import requests


# ============================================================
# Config
# ============================================================

SERVER_URL = "http://127.0.0.1:8001/predict"

VIDEO_PATH = "datasets/action/real_case/0_3.mp4"

ACTION_JSON = (
    "datasets/action/real_case/"
    "nav_straight_backward_anchored_60.json"
)

OUTPUT_VIDEO = "outputs/fd_server_test.mp4"

PROMPT = "You are an autonomous vehicle planning system."

DOMAIN_NAME = "av"

IMAGE_SIZE = 480


# ============================================================
# 1. 从视频读取第一帧
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame_bgr = cap.read()

cap.release()

if not ret:
    raise RuntimeError(
        f"Cannot read first frame from {VIDEO_PATH}"
    )

print(
    "input frame:",
    frame_bgr.shape,
    frame_bgr.dtype,
)


# ============================================================
# 2. OpenCV BGR -> RGB -> PNG -> base64
# ============================================================

frame_rgb = cv2.cvtColor(
    frame_bgr,
    cv2.COLOR_BGR2RGB,
)

# 注意：
# cv2.imencode期待BGR也没关系，因为这里我们直接用原始BGR编码，
# 服务端PIL读PNG后会得到正确RGB。
ok, encoded = cv2.imencode(
    ".png",
    frame_bgr,
)

if not ok:
    raise RuntimeError(
        "Failed to encode input frame."
    )

image_b64 = base64.b64encode(
    encoded.tobytes()
).decode("ascii")


# ============================================================
# 3. 加载 60x9 action
# ============================================================

with open(ACTION_JSON, "r") as f:
    action = np.asarray(
        json.load(f),
        dtype=np.float32,
    )

print(
    "action:",
    action.shape,
    action.dtype,
)

if action.shape != (60, 9):
    raise ValueError(
        f"Expected action shape (60,9), "
        f"got {action.shape}"
    )


# ============================================================
# 4. HTTP request
# ============================================================

payload = {
    "image": image_b64,
    "prompt": PROMPT,
    "domain_name": DOMAIN_NAME,
    "image_size": IMAGE_SIZE,

    # numpy不能直接json序列化
    "action": action.tolist(),
}


print("sending request to:", SERVER_URL)

response = requests.post(
    SERVER_URL,
    json=payload,

    # Cosmos视频生成可能比较慢
    timeout=600,
)

print(
    "HTTP status:",
    response.status_code,
)

if response.status_code != 200:
    print(response.text)
    raise RuntimeError(
        "Server request failed."
    )

result = response.json()

print(
    "response keys:",
    result.keys(),
)

if "error" in result:
    raise RuntimeError(
        result["error"]
    )


# ============================================================
# 5. 解码返回的 base64 frames
# ============================================================

video_b64 = result["video"]

print(
    "num returned frames:",
    len(video_b64),
)

frames = []

for i, frame_str in enumerate(video_b64):

    raw = base64.b64decode(
        frame_str
    )

    arr = np.frombuffer(
        raw,
        dtype=np.uint8,
    )

    frame_bgr = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR,
    )

    if frame_bgr is None:
        raise RuntimeError(
            f"Failed to decode returned frame {i}"
        )

    frames.append(
        frame_bgr
    )


# ============================================================
# 6. 保存 mp4
# ============================================================

fps = result.get(
    "fps",
    10,
)

output_path = Path(
    OUTPUT_VIDEO
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

H, W = frames[0].shape[:2]

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    float(fps),
    (W, H),
)

if not writer.isOpened():
    raise RuntimeError(
        f"Cannot create {output_path}"
    )

for frame in frames:
    writer.write(frame)

writer.release()

print(
    "saved video:",
    output_path
)