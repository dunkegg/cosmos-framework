import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


# ============================================================
# Config
# ============================================================

QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8002")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "/mnt/ws_nas/models/Qwen3.8-27B")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "EMPTY")

COSMOS_SERVER_URL = os.environ.get("COSMOS_SERVER_URL", "http://127.0.0.1:8001/predict")

VIDEO_PATH = "datasets/action/real_case/indoor_f.mp4"
REFERENCE_ACTION_JSON = "datasets/action/real_case/nav_straight_backward_anchored_60.json"

OUTPUT_ACTION_JSON = "outputs/qwen_generated_action_60x9.json"
OUTPUT_VIDEO = "outputs/qwen_cosmos_plan.mp4"

DOMAIN_NAME = "av"
IMAGE_SIZE = 480
ACTION_CHUNK_SIZE = 60
FPS = 10

COSMOS_PROMPT = "You are an autonomous vehicle planning system."

REWARD_QWEN_BASE_URL = os.environ.get("REWARD_QWEN_BASE_URL", QWEN_BASE_URL)
REWARD_QWEN_MODEL = os.environ.get("REWARD_QWEN_MODEL", QWEN_MODEL)
REWARD_QWEN_API_KEY = os.environ.get("REWARD_QWEN_API_KEY", QWEN_API_KEY)

OUTPUT_REWARD_JSON = "outputs/qwen_cosmos_progress_reward.json"
CANDIDATE_OUTPUT_DIR = "outputs/candidate_reward_test_right"
SLIGHT_TURN_YAW_DEG = 15.0
REWARD_ROLLOUT_FRAME_INDICES = [10, 20, 30, 40, 50, 60]


# ============================================================
# Video
# ============================================================

def read_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read first frame from: {video_path}")

    print("input frame:", frame_bgr.shape, frame_bgr.dtype)
    return frame_bgr


def encode_image_base64(frame_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", frame_bgr)

    if not ok:
        raise RuntimeError("Failed to encode frame.")

    return base64.b64encode(encoded.tobytes()).decode("ascii")


# ============================================================
# Qwen
# ============================================================

PLANNER_SYSTEM_PROMPT = """
You are a high-level motion planner for an autonomous mobile robot
or autonomous vehicle.

Convert the user's navigation instruction into ONE short motion plan.

Return ONLY valid JSON.
Do not use Markdown.
Do not output explanations outside the JSON.

Use exactly this schema:

{
  "motion": "stop | forward | backward | left | right | forward_left | forward_right",
  "distance_m": number,
  "yaw_deg": number,
  "speed_mps": number,
  "reason": string
}

Coordinate convention for the high-level plan:

- forward/backward are semantic navigation directions, not Cosmos tensor axes
- positive yaw_deg means turning LEFT
- negative yaw_deg means turning RIGHT
- straight forward/backward should normally have yaw_deg = 0
- stop should have distance_m = 0 and speed_mps = 0

The downstream controller maps navigation coordinates into Cosmos AV coordinates:
- Cosmos translation x = robot RIGHT
- Cosmos translation y = vertical (0 for planar navigation)
- Cosmos translation z = robot FORWARD

The downstream controller converts the plan into a 60-step
backward-anchored SE(2) trajectory.

Use conservative movements.
""".strip()


def qwen_chat(
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    url = f"{QWEN_BASE_URL.rstrip('/')}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
    }

    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Raw HTTP body for vLLM. If the model/template ignores this field,
        # extract_json_object() and retry logic still keep the client robust.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    print("sending Qwen request to:", url)

    response = requests.post(url, headers=headers, json=payload, timeout=(10, 300))

    print("Qwen HTTP status:", response.status_code)

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Robustly extract one JSON object from LLM/VLM output.

    Handles:
    - <think>...</think>
    - markdown ```json ... ```
    - extra text before/after JSON
    - BOM
    """

    if not isinstance(text, str):
        raise TypeError(
            f"Expected str, got {type(text)}"
        )

    text = text.strip()

    # --------------------------------------------------
    # 1. 去 BOM
    # --------------------------------------------------

    text = text.lstrip("\ufeff")

    # --------------------------------------------------
    # 2. 去掉 Qwen reasoning
    #
    # 例如：
    #
    # <think>
    # ...
    # </think>
    #
    # {"motion": ...}
    # --------------------------------------------------

    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # 如果只有残留标签，也一起去掉
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    text = text.strip()

    # --------------------------------------------------
    # 3. 去 Markdown fence
    #
    # ```json
    # {...}
    # ```
    # --------------------------------------------------

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    text = text.strip()

    # --------------------------------------------------
    # 4. 最理想情况：整个字符串就是 JSON
    # --------------------------------------------------

    try:

        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------
    # 5. 从脏文本中提取 {...}
    #
    # 比如：
    #
    # Sure, here is the result:
    #
    # {
    #   "motion": "forward"
    # }
    #
    # Hope this helps.
    # --------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        candidate = text[
            start:end + 1
        ].strip()

        try:

            obj = json.loads(candidate)

            if isinstance(obj, dict):
                return obj

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------
    # 6. 最后一层：逐字符寻找第一个可解析 JSON object
    #
    # 可以处理：
    #
    # {...} some text {...}
    # --------------------------------------------------

    decoder = json.JSONDecoder()

    for i, ch in enumerate(text):

        if ch != "{":
            continue

        try:

            obj, _ = decoder.raw_decode(
                text[i:]
            )

            if isinstance(obj, dict):
                return obj

        except json.JSONDecodeError:
            continue

    # --------------------------------------------------
    # 全部失败
    # --------------------------------------------------

    raise RuntimeError(
        "Cannot extract valid JSON object.\n"
        f"Raw output:\n{text}"
    )


def qwen_plan(
    instruction: str,
    max_retry: int = 2,
) -> dict[str, Any]:
    """Parse the navigation instruction once into a target motion specification."""
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    last_content = None
    last_error = None

    for attempt in range(max_retry + 1):
        content = qwen_chat(messages, temperature=0.0, max_tokens=256)
        last_content = content
        print(f"\nQwen planner raw output (attempt {attempt + 1}):")
        print(content)

        try:
            plan = extract_json_object(content)
            required = ["motion", "distance_m", "yaw_deg", "speed_mps"]
            for key in required:
                if key not in plan:
                    raise RuntimeError(f"Planner missing key: {key}")

            plan["motion"] = str(plan["motion"]).strip().lower()
            plan["distance_m"] = abs(float(plan["distance_m"]))
            plan["yaw_deg"] = float(plan["yaw_deg"])
            plan["speed_mps"] = max(0.0, float(plan["speed_mps"]))
            plan["reason"] = str(plan.get("reason", ""))

            print("parsed target plan:")
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return plan

        except Exception as e:
            last_error = e
            print(f"Planner parse failed ({attempt + 1}/{max_retry + 1}): {e}")
            if attempt >= max_retry:
                break

            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Return ONLY one valid JSON object using the required schema. "
                            "No reasoning, Markdown, code fences, or text outside JSON."
                        ),
                    },
                ]
            )

    raise RuntimeError(
        "Qwen planner failed after retries.\n"
        f"Last error: {last_error}\n"
        f"Last output:\n{last_content}"
    )


# ============================================================
# Rotation
# ============================================================

def yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    """
    Cosmos AV convention used here:
      x = right
      y = vertical
      z = forward

    Planar yaw therefore rotates around the vertical Y axis.
    """
    c = math.cos(yaw)
    s = math.sin(yaw)

    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ], dtype=np.float32)


def rotation_matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """
    rot6d = first two columns of rotation matrix.

    Identity:
        [1, 0, 0, 0, 1, 0]
    """
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


# ============================================================
# Plan -> 60x9 backward_anchored action
# ============================================================

def plan_to_action(plan: dict[str, Any], num_steps: int = 60) -> np.ndarray:
    """
    Navigation:
      forward = +nav_x
      right   = +nav_y

    Cosmos AV translation:
      action[..., 0] = right
      action[..., 1] = vertical
      action[..., 2] = forward
    """
    motion = str(plan["motion"]).lower()
    distance = abs(float(plan["distance_m"]))
    yaw_deg = float(plan["yaw_deg"])

    if motion == "stop":
        distance = 0.0
        yaw_deg = 0.0
    elif motion == "forward":
        yaw_deg = 0.0
    elif motion == "backward":
        distance = -distance
        yaw_deg = 0.0
    elif motion == "left":
        yaw_deg = abs(yaw_deg) if abs(yaw_deg) > 1e-4 else 30.0
    elif motion == "right":
        yaw_deg = -abs(yaw_deg) if abs(yaw_deg) > 1e-4 else -30.0
    elif motion == "forward_left":
        distance = abs(distance)
        yaw_deg = abs(yaw_deg) if abs(yaw_deg) > 1e-4 else 15.0
    elif motion == "forward_right":
        distance = abs(distance)
        yaw_deg = -abs(yaw_deg) if abs(yaw_deg) > 1e-4 else -15.0
    else:
        raise ValueError(f"Unsupported motion: {motion}")

    yaw_deg = yaw_deg*0.4
    total_yaw = math.radians(yaw_deg)
    action = np.zeros((num_steps, 9), dtype=np.float32)

    for i in range(num_steps):
        alpha = (i + 1) / num_steps
        current_distance = distance * alpha
        current_yaw = total_yaw * alpha

        if abs(total_yaw) < 1e-6:
            nav_forward = current_distance
            nav_right = 0.0
        elif abs(distance) < 1e-6:
            nav_forward = 0.0
            nav_right = 0.0
        else:
            radius = distance / total_yaw
            nav_forward = radius * math.sin(current_yaw)
            # positive planner yaw = LEFT -> nav_right negative
            nav_right = -radius * (1.0 - math.cos(current_yaw))

        tx = nav_right
        ty = 0.0
        tz = nav_forward

        # positive planner yaw means LEFT; negate for this Y-axis convention
        R = yaw_to_rotation_matrix(-current_yaw)
        rot6d = rotation_matrix_to_rot6d(R)

        action[i, :3] = [tx, ty, tz]
        action[i, 3:] = rot6d

    return action


# ============================================================
# Action utilities
# ============================================================

def save_action_json(action: np.ndarray, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(action.tolist(), f, indent=2)

    print("saved action:", path)


def print_action_summary(action: np.ndarray):
    print("action shape:", action.shape)
    print("first:", action[0])
    print("second:", action[1])
    print("last:", action[-1])
    print("final translation:", action[-1, :3])
    print("final rot6d:", action[-1, 3:])


def compare_reference_action(generated: np.ndarray, reference_path: str):
    path = Path(reference_path)

    if not path.exists():
        print("reference action does not exist:", path)
        return

    with open(path, "r") as f:
        reference = np.asarray(json.load(f), dtype=np.float32)

    print("reference shape:", reference.shape)
    print("generated shape:", generated.shape)

    if reference.shape != generated.shape:
        print("shape mismatch, skip comparison")
        return

    diff = np.abs(generated - reference)

    print("mean abs diff:", diff.mean())
    print("max abs diff:", diff.max())
    print("reference first:", reference[0])
    print("generated first:", generated[0])
    print("reference last:", reference[-1])
    print("generated last:", generated[-1])


# ============================================================
# Cosmos
# ============================================================

def cosmos_predict(frame_bgr: np.ndarray, action: np.ndarray) -> tuple[list[np.ndarray], float]:
    if action.shape != (ACTION_CHUNK_SIZE, 9):
        raise ValueError(f"Expected action shape ({ACTION_CHUNK_SIZE}, 9), got {action.shape}")

    payload = {
        "image": encode_image_base64(frame_bgr),
        "prompt": COSMOS_PROMPT,
        "domain_name": DOMAIN_NAME,
        "image_size": IMAGE_SIZE,
        "action": action.tolist(),
    }

    print("sending Cosmos request to:", COSMOS_SERVER_URL)

    response = requests.post(
        COSMOS_SERVER_URL,
        json=payload,
        timeout=(10, 1800),
    )

    print("Cosmos HTTP status:", response.status_code)

    if response.status_code != 200:
        print(response.text)
        raise RuntimeError("Cosmos request failed.")

    result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])

    frames = []

    for i, frame_str in enumerate(result["video"]):
        raw = base64.b64decode(frame_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError(f"Failed to decode Cosmos frame {i}")

        frames.append(frame)

    fps = float(result.get("fps", FPS))

    print("returned frames:", len(frames))
    print("returned fps:", fps)

    return frames, fps


# ============================================================
# Action-geometry reward
# ============================================================

def instruction_has_visual_target(instruction: str) -> bool:
    """
    Lightweight first-pass detector. This only decides whether visibility gain
    should participate in reward; it does not parse the whole instruction.
    """
    s = instruction.lower()
    target_terms = [
        "find ", "look for", "approach ", "go to ", "reach ",
        "door", "chair", "person", "pedestrian", "car", "truck",
        "elevator", "sign", "cone", "object", "target", "landmark",
    ]
    return any(term in s for term in target_terms)


def normalized_motion_target(plan: dict[str, Any]) -> tuple[float, float]:
    """
    Convert high-level plan to target signed distance and yaw in degrees.
    """
    motion = str(plan["motion"]).lower()
    distance = abs(float(plan["distance_m"]))
    yaw = float(plan["yaw_deg"])

    if motion == "stop":
        return 0.0, 0.0
    if motion == "backward":
        return -distance, 0.0
    if motion == "forward":
        return distance, 0.0
    if motion == "left":
        return 0.0, abs(yaw) if abs(yaw) > 1e-4 else 30.0
    if motion == "right":
        return 0.0, -abs(yaw) if abs(yaw) > 1e-4 else -30.0
    if motion == "forward_left":
        return distance, abs(yaw) if abs(yaw) > 1e-4 else 15.0
    if motion == "forward_right":
        return distance, -abs(yaw) if abs(yaw) > 1e-4 else -15.0
    raise ValueError(f"Unsupported motion: {motion}")


def action_geometry_reward(
    candidate_plan: dict[str, Any],
    target_plan: dict[str, Any],
    action: np.ndarray,
) -> dict[str, float]:
    """
    Deterministic reward for quantities already known from the action plan.

    VLM should not guess exact meters/yaw from monocular generated video.
    """
    cand_distance, cand_yaw = normalized_motion_target(candidate_plan)
    target_distance, target_yaw = normalized_motion_target(target_plan)

    # Distance tolerance scales with commanded distance, with a 0.15 m floor.
    dist_scale = max(abs(target_distance), 0.15)
    distance_error = abs(cand_distance - target_distance)
    distance_correctness = float(np.clip(1.0 - distance_error / dist_scale, 0.0, 1.0))

    # 15 deg error should already be meaningfully penalized for this local test.
    yaw_scale = 15.0
    yaw_error = abs(cand_yaw - target_yaw)
    yaw_correctness = float(np.clip(1.0 - yaw_error / yaw_scale, 0.0, 1.0))

    # Smoothness from first differences of anchored translation and rot6d.
    if len(action) > 1:
        delta = np.diff(action, axis=0)
        transl_jerk_proxy = float(np.mean(np.linalg.norm(np.diff(delta[:, :3], axis=0), axis=1))) if len(delta) > 1 else 0.0
        rot_jerk_proxy = float(np.mean(np.linalg.norm(np.diff(delta[:, 3:], axis=0), axis=1))) if len(delta) > 1 else 0.0
    else:
        transl_jerk_proxy = 0.0
        rot_jerk_proxy = 0.0

    smoothness = float(np.exp(-20.0 * transl_jerk_proxy - 5.0 * rot_jerk_proxy))

    action_reward = (
        0.45 * distance_correctness
        + 0.45 * yaw_correctness
        + 0.10 * smoothness
    )

    return {
        "distance_correctness": distance_correctness,
        "yaw_correctness": yaw_correctness,
        "smoothness": smoothness,
        "distance_error_m": float(distance_error),
        "yaw_error_deg": float(yaw_error),
        "action_reward": float(action_reward),
    }


# ============================================================
# Progress reward
# ============================================================

PROGRESS_REWARD_SYSTEM_PROMPT = """
You are a navigation progress evaluator for an embodied mobile agent.

You are given:
1. A navigation instruction.
2. The current observation before executing an action.
3. Several temporally ordered frames from a predicted future rollout.

Evaluate ONLY what can be judged reliably from visual temporal evidence.
Do NOT estimate exact metric travel distance or exact yaw angle from the images;
those are evaluated separately from the known action trajectory.

Evaluate:
- progress: -1..1, visual semantic progress toward the instruction
- subgoal_progress: 0..1
- direction_correctness: 0..1
- route_consistency: 0..1
- collision_risk: 0..1
- regression: 0..1
- confidence: 0..1

For target_visibility_gain:
- if the instruction explicitly mentions a visual target/landmark/object, score 0..1
- otherwise return exactly 0.5 as a neutral value

Ignore pure image-generation quality unless artifacts make navigation impossible to judge.
Do not describe every frame. Return ONLY valid JSON:

{
  "progress": number,
  "subgoal_progress": number,
  "direction_correctness": number,
  "target_visibility_gain": number,
  "route_consistency": number,
  "collision_risk": number,
  "regression": number,
  "confidence": number,
  "reason": string
}
""".strip()


def frame_bgr_to_data_url(frame_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode frame for reward evaluator.")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def qwen_progress_reward(
    instruction: str,
    current_frame_bgr: np.ndarray,
    cosmos_frames_bgr: list[np.ndarray],
) -> dict[str, Any]:
    if not cosmos_frames_bgr:
        raise ValueError("No Cosmos rollout frames for reward evaluation.")

    used_indices = []
    sampled = []
    for idx in REWARD_ROLLOUT_FRAME_INDICES:
        real_idx = min(max(int(idx), 0), len(cosmos_frames_bgr) - 1)
        if real_idx in used_indices:
            continue
        used_indices.append(real_idx)
        sampled.append(cosmos_frames_bgr[real_idx])

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Navigation instruction:\n{instruction}\n\n"
                "The next image is the CURRENT observation before the action."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": frame_bgr_to_data_url(current_frame_bgr)},
        },
        {
            "type": "text",
            "text": "The following images are predicted FUTURE rollout frames in chronological order.",
        },
    ]

    for i, frame in enumerate(sampled):
        content.append({"type": "text", "text": f"Future frame {i+1}/{len(sampled)}:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame_bgr_to_data_url(frame)},
            }
        )

    messages = [
        {"role": "system", "content": PROGRESS_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    url = f"{REWARD_QWEN_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {REWARD_QWEN_API_KEY}",
    }
    payload = {
        "model": REWARD_QWEN_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,

        "chat_template_kwargs": {"enable_thinking": False},

    }

    print("sending progress reward request to:", url)
    print("reward rollout indices:", used_indices)

    response = requests.post(url, headers=headers, json=payload, timeout=(10, 600))
    print("Progress reward HTTP status:", response.status_code)

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"]
    print("Progress reward raw output:")
    print(raw)

    score = extract_json_object(raw)

    required = [
        "progress",
        "subgoal_progress",
        "direction_correctness",
        "target_visibility_gain",
        "route_consistency",
        "collision_risk",
        "regression",
        "confidence",
    ]
    for key in required:
        if key not in score:
            raise RuntimeError(f"Progress reward missing key: {key}")
        score[key] = float(score[key])

    score["progress"] = float(np.clip(score["progress"], -1.0, 1.0))
    for key in required[1:]:
        score[key] = float(np.clip(score[key], 0.0, 1.0))

    has_visual_target = instruction_has_visual_target(instruction)

    # Vision reward only uses quantities the VLM can judge reliably.
    positive_terms = [
        (0.40, score["subgoal_progress"]),
        (0.25, score["direction_correctness"]),
        (0.20, score["route_consistency"]),
    ]

    if has_visual_target:
        positive_terms.append((0.15, score["target_visibility_gain"]))

    positive_weight = sum(w for w, _ in positive_terms)
    positive_reward = sum(w * v for w, v in positive_terms) / max(positive_weight, 1e-6)

    visual_reward = (
        positive_reward
        - 0.40 * score["collision_risk"]
        - 0.25 * score["regression"]
    )

    score["has_visual_target"] = bool(has_visual_target)
    score["visual_reward"] = float(visual_reward)
    score["confidence_weighted_visual_reward"] = float(visual_reward * score["confidence"])
    score["rollout_frame_indices"] = used_indices

    print("parsed progress reward:")
    print(json.dumps(score, indent=2, ensure_ascii=False))
    return score


def save_reward_json(reward: dict[str, Any], output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(reward, f, indent=2, ensure_ascii=False)
    print("saved reward:", path)


# ============================================================
# Save video
# ============================================================

def save_video(frames: list[np.ndarray], fps: float, output_path: str):
    if not frames:
        raise ValueError("No frames to save.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

    writer.release()

    print("saved video:", path)


# ============================================================
# Three-candidate reward test
# ============================================================

def make_test_candidates(distance_m: float = 1.2) -> list[dict[str, Any]]:
    """
    Minimal sanity-test candidates for one current observation.

    All three travel approximately the same arc length:
      1) straight
      2) slight left
      3) slight right

    The left/right magnitude is intentionally small so Cosmos stays close to the
    stable forward-motion distribution.
    """
    yaw = float(SLIGHT_TURN_YAW_DEG)

    return [
        {
            "name": "straight",
            "plan": {
                "motion": "forward",
                "distance_m": float(distance_m),
                "yaw_deg": 0.0,
                "speed_mps": float(distance_m) / (ACTION_CHUNK_SIZE / FPS),
                "reason": "Straight baseline candidate.",
            },
        },
        {
            "name": "slight_left",
            "plan": {
                "motion": "forward_left",
                "distance_m": float(distance_m),
                "yaw_deg": +yaw,
                "speed_mps": float(distance_m) / (ACTION_CHUNK_SIZE / FPS),
                "reason": f"Slight left candidate with {yaw:.1f} degree total yaw.",
            },
        },
        {
            "name": "slight_right",
            "plan": {
                "motion": "forward_right",
                "distance_m": float(distance_m),
                "yaw_deg": -yaw,
                "speed_mps": float(distance_m) / (ACTION_CHUNK_SIZE / FPS),
                "reason": f"Slight right candidate with {yaw:.1f} degree total yaw.",
            },
        },
    ]


def run_three_candidate_reward_test(
    instruction: str,
    frame_bgr: np.ndarray,
) -> dict[str, Any]:
    output_dir = Path(CANDIDATE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    # Parse the instruction ONCE. This gives us exact target distance/yaw for the
    # deterministic action reward; the VLM is used only for visual semantics.
    target_plan = qwen_plan(instruction)

    candidates = make_test_candidates(distance_m=float(target_plan["distance_m"]))

    print("\n" + "=" * 72)
    print("THREE-CANDIDATE WORLD-MODEL REWARD TEST")
    print("=" * 72)
    print("instruction:", instruction)
    print("candidates:", [c["name"] for c in candidates])
    print("slight turn yaw:", SLIGHT_TURN_YAW_DEG, "deg")

    for candidate in candidates:
        name = candidate["name"]
        plan = candidate["plan"]

        print("\n" + "-" * 72)
        print("candidate:", name)
        print(json.dumps(plan, indent=2, ensure_ascii=False))

        action = plan_to_action(plan, num_steps=ACTION_CHUNK_SIZE)
        print_action_summary(action)

        action_path = output_dir / f"{name}_action_60x9.json"
        video_path = output_dir / f"{name}_rollout.mp4"
        reward_path = output_dir / f"{name}_reward.json"

        save_action_json(action, str(action_path))

        # All candidates use the same current observation and the Cosmos server's
        # fixed seed, so the main controlled variable is the action trajectory.
        frames, fps = cosmos_predict(frame_bgr, action)
        save_video(frames, fps, str(video_path))

        visual_reward = qwen_progress_reward(
            instruction=instruction,
            current_frame_bgr=frame_bgr,
            cosmos_frames_bgr=frames,
        )

        action_reward = action_geometry_reward(
            candidate_plan=plan,
            target_plan=target_plan,
            action=action,
        )

        # Final test reward:
        #   65% visual world-model evidence
        #   35% exact action-geometry compliance
        #
        # This keeps the world model central while avoiding asking the VLM to
        # infer exact meters/yaw from monocular generated video.
        selection_reward = (
            0.65 * visual_reward["confidence_weighted_visual_reward"]
            + 0.35 * action_reward["action_reward"]
        )

        reward = {
            "visual": visual_reward,
            "action_geometry": action_reward,
            "selection_reward": float(selection_reward),
        }
        save_reward_json(reward, str(reward_path))

        result = {
            "name": name,
            "plan": plan,
            "selection_reward": float(selection_reward),
            "progress": float(visual_reward["progress"]),
            "confidence": float(visual_reward["confidence"]),
            "subgoal_progress": float(visual_reward["subgoal_progress"]),
            "direction_correctness": float(visual_reward["direction_correctness"]),
            "route_consistency": float(visual_reward["route_consistency"]),
            "collision_risk": float(visual_reward["collision_risk"]),
            "regression": float(visual_reward["regression"]),
            "visual_reward": float(visual_reward["confidence_weighted_visual_reward"]),
            "action_reward": float(action_reward["action_reward"]),
            "distance_correctness": float(action_reward["distance_correctness"]),
            "yaw_correctness": float(action_reward["yaw_correctness"]),
            "reason": visual_reward.get("reason", ""),
            "action_path": str(action_path),
            "video_path": str(video_path),
            "reward_path": str(reward_path),
        }
        results.append(result)

    # Higher is better.
    ranking = sorted(
        results,
        key=lambda x: x["selection_reward"],
        reverse=True,
    )
    winner = ranking[0]

    summary = {
        "instruction": instruction,
        "selection_metric": "0.65 * confidence_weighted_visual_reward + 0.35 * action_reward",
        "target_plan": target_plan,
        "slight_turn_yaw_deg": float(SLIGHT_TURN_YAW_DEG),
        "ranking": ranking,
        "winner": winner["name"],
        "winner_reward": winner["selection_reward"],
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("FINAL RANKING")
    print("=" * 72)

    for rank_idx, item in enumerate(ranking, start=1):
        print(
            f"{rank_idx}. {item['name']:>12s} | "
            f"total={item['selection_reward']:+.4f} | "
            f"vision={item['visual_reward']:+.4f} | "
            f"action={item['action_reward']:+.4f} | "
            f"progress={item['progress']:+.3f} | "
            f"dir={item['direction_correctness']:.3f} | "
            f"yaw_ok={item['yaw_correctness']:.3f} | "
            f"dist_ok={item['distance_correctness']:.3f} | "
            f"collision={item['collision_risk']:.3f}"
        )

    print("\nWINNER:", winner["name"])
    print("WINNER REWARD:", winner["selection_reward"])
    print("summary:", summary_path)
    print("No action is executed; this is evaluation-only.")

    return summary


# ============================================================
# Main
# ============================================================

def main():
    # Sanity-test instruction: because it explicitly requests straight motion,
    # a healthy evaluator should normally rank "straight" above slight-left/right.
    # instruction = "Drive straight forward for 1.2 meters without turning."
    instruction = "Drive forward and turn slightly right."

    print("instruction:", instruction)

    frame_bgr = read_first_frame(VIDEO_PATH)

    # No execution. Generate three candidate rollouts, evaluate each predicted
    # future, rank them by progress reward, and print the winner.
    run_three_candidate_reward_test(
        instruction=instruction,
        frame_bgr=frame_bgr,
    )


if __name__ == "__main__":
    main()