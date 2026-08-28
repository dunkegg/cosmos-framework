import json
import os
from typing import Any

import requests


# ============================================================
# Config
# ============================================================

QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "http://127.0.0.1:8002",
)

QWEN_MODEL = os.environ.get(
    "QWEN_MODEL",
    "/mnt/ws_nas/models/Qwen3.8-27B",
)

QWEN_API_KEY = os.environ.get(
    "QWEN_API_KEY",
    "EMPTY",
)

REQUEST_TIMEOUT = float(
    os.environ.get("QWEN_REQUEST_TIMEOUT", "300")
)


# ============================================================
# Low-level vLLM client
# ============================================================

def qwen_chat(
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Call vLLM's OpenAI-compatible /v1/chat/completions endpoint.
    """

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
    }

    print("sending request to:", url)
    print("model:", QWEN_MODEL)

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(10, REQUEST_TIMEOUT),
    )

    if response.status_code != 200:
        print("HTTP status:", response.status_code)
        print(response.text)
        response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"]


# ============================================================
# Structured planner for AV / Cosmos
# ============================================================

PLANNER_SYSTEM_PROMPT = """
You are an autonomous vehicle high-level motion planner.

Your output will be converted by another program into a 60x9
Cosmos forward-dynamics action trajectory.

Return ONLY one valid JSON object.
Do not return Markdown.
Do not return explanations outside the JSON.

Schema:
{
  "motion": "stop | forward | backward | left | right | forward_left | forward_right",
  "distance_m": number,
  "yaw_deg": number,
  "speed_mps": number,
  "reason": string
}

Rules:
- yaw_deg > 0 means left turn.
- yaw_deg < 0 means right turn.
- For straight motion, yaw_deg should be 0.
- For stop, distance_m and speed_mps should be 0.
- Prefer conservative vehicle motions.
""".strip()


def qwen_plan(
    instruction: str,
) -> dict[str, Any]:
    """
    Convert a natural-language navigation instruction to a structured plan.
    """

    content = qwen_chat(
        messages=[
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": instruction,
            },
        ],
        temperature=0.1,
        max_tokens=256,
    )

    print("\nraw model output:")
    print(content)

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Qwen did not return valid JSON.\n"
            f"Raw output:\n{content}"
        ) from exc

    required_keys = {
        "motion",
        "distance_m",
        "yaw_deg",
        "speed_mps",
    }

    missing = required_keys - set(plan.keys())

    if missing:
        raise RuntimeError(
            f"Missing keys in planner output: {sorted(missing)}"
        )

    return plan


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":
    instruction = (
        "Drive straight forward for approximately 2 meters "
        "without turning."
    )

    plan = qwen_plan(instruction)

    print("\nparsed plan:")
    print(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        )
    )