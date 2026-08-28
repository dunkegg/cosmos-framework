# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Multi-GPU HTTP inference server for Cosmos3 AV Forward Dynamics.

8-rank architecture:

    rank0:
        HTTP request
            |
            v
        broadcast request
            |
    +-------+-------+-------+
    |       |       |       |
  rank0   rank1   ...     rank7
    |       |               |
    +---- Cosmos FD --------+
            |
            v
         rank0
            |
            v
      HTTP response

Request:

POST /predict

{
    "image": "<base64 PNG>",
    "prompt": "You are an autonomous vehicle planning system.",
    "domain_name": "av",
    "image_size": 480,
    "action": [
        [a00, ..., a08],
        ...
        [a590, ..., a598]
    ]
}

action shape:
    [60, 9]

Response:

{
    "video": ["<base64 PNG>", ...],
    "num_frames": 61,
    "seed": 0
}
"""

from __future__ import annotations

# IMPORTANT:
# init_script must run before most cosmos imports.

from cosmos_framework.inference.common.init import (
    init_script,
    is_rank0,
)

init_script()

import base64
import io
import json
import threading
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import time
import numpy as np
import pydantic
import torch
import torch.distributed as dist
import tyro
from PIL import Image


# ============================================================
# Cosmos imports
# ============================================================

from cosmos_framework.data.generator.action.action_processing import ActionProcessingRecord, make_batched_action_processing_fields

from cosmos_framework.data.generator.action.domain_utils import (
    get_domain_id,
)

from cosmos_framework.data.generator.action.json_formatter import (
    ActionPromptJsonFormatter,
)

from cosmos_framework.data.generator.action.transforms import build_sequence_plan_from_mode, find_closest_target_size, reflection_pad_to_target, remove_reflection_padding

from cosmos_framework.inference.args import (
    OmniSetupArgs,
    OmniSetupOverrides,
)

from cosmos_framework.inference.common.args import (
    CheckpointOverrides,
    tyro_cli,
)

from cosmos_framework.inference.common.init import init_output_dir

from cosmos_framework.inference.inference import OmniInference

from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)

from cosmos_framework.utils import log

from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution


# ============================================================
# Defaults
# ============================================================

_DEFAULT_ACTION_CHUNK_SIZE = 60

_DEFAULT_RAW_ACTION_DIM = 9

_DEFAULT_FPS = 10

_DEFAULT_DOMAIN = "av"

_DEFAULT_PROMPT = (
    "You are an autonomous vehicle planning system."
)


# ============================================================
# Distributed helpers
# ============================================================


def _dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _rank() -> int:
    return dist.get_rank() if _dist_enabled() else 0


def _world_size() -> int:
    return dist.get_world_size() if _dist_enabled() else 1


def _local_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


# ============================================================
# Image helpers
# ============================================================


def _decode_base64_png(
    image_b64: str,
) -> torch.Tensor:
    """
    base64 PNG -> RGB uint8 [3,H,W]
    """

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    raw = base64.b64decode(image_b64)

    with Image.open(io.BytesIO(raw)) as img:

        img = img.convert("RGB")

        arr = np.asarray(
            img,
            dtype=np.uint8,
        ).copy()

    tensor = (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .contiguous()
    )

    return tensor


def _encode_video_frames(
    video_c_t_h_w: torch.Tensor,
) -> list[str]:
    """
    [C,T,H,W] float tensor
            ->
    list of base64 PNG
    """

    video = (
        video_c_t_h_w
        .detach()
        .cpu()
        .float()
    )

    if video.min() < 0:

        video = (
            video + 1.0
        ) / 2.0

    video = video.clamp(
        0.0,
        1.0,
    )

    output = []

    T = int(video.shape[1])

    for t in range(T):

        frame = (
            video[:, t]
            * 255.0
        ).round().to(torch.uint8)

        frame = (
            frame
            .permute(1, 2, 0)
            .numpy()
        )

        pil = Image.fromarray(frame)

        buf = io.BytesIO()

        pil.save(
            buf,
            format="PNG",
        )

        output.append(
            base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
        )

    return output


# ============================================================
# Prompt helper
# ============================================================


def _augment_prompt(
    prompt: str,
    *,
    t_frames: int,
    fps: int,
    height: int,
    width: int,
) -> str:

    duration = (
        float(t_frames) / float(fps)
    )

    return (
        f"{prompt} "
        f"The video is {duration:.1f} seconds long "
        f"and is of {fps} FPS. "
        f"This video is of "
        f"{height}x{width} resolution."
    )


# ============================================================
# CLI args
# ============================================================


class ForwardDynamicsServerArgs(
    pydantic.BaseModel
):

    model_config = pydantic.ConfigDict(
        extra="forbid",
        use_attribute_docstrings=True,
    )

    checkpoint: tyro.conf.OmitArgPrefixes[
        CheckpointOverrides
    ] = CheckpointOverrides.model_construct()

    output_dir: Path | None = None

    # --------------------------------------------------------
    # distributed
    # --------------------------------------------------------

    parallelism_preset: str = "throughput"

    # --------------------------------------------------------
    # server
    # --------------------------------------------------------

    host: str = "0.0.0.0"

    port: int = 8001

    # --------------------------------------------------------
    # inference
    # --------------------------------------------------------

    sampler: Literal[
        "unipc",
        "edm",
    ] = "unipc"

    seed: int = 0

    guidance: float = 1.0

    num_steps: int = 30

    fps: int = _DEFAULT_FPS

    action_chunk_size: int = (
        _DEFAULT_ACTION_CHUNK_SIZE
    )

    raw_action_dim: int = (
        _DEFAULT_RAW_ACTION_DIM
    )

    domain_name: str = (
        _DEFAULT_DOMAIN
    )

    image_size: int = 480

    format_prompt_as_json: bool = False

    def build_setup_overrides(
        self,
    ) -> OmniSetupOverrides:

        if not getattr(
            self.checkpoint,
            "checkpoint_path",
            "",
        ):
            raise ValueError(
                "--checkpoint-path is required"
            )

        base = (
            OmniSetupOverrides.model_validate(
                self.checkpoint.model_dump()
            )
        )

        base.output_dir = (
            self.output_dir
            or DEFAULT_FALLBACK_OUTPUT_DIR
            / "fd_av_server"
        )

        base.sampler = self.sampler

        # ----------------------------------------------
        # Keep identical to:
        #
        # --parallelism-preset=throughput
        # ----------------------------------------------

        if (
            "parallelism_preset"
            in type(base).model_fields
        ):
            base.parallelism_preset = (
                self.parallelism_preset
            )

        else:
            raise RuntimeError(
                "This Cosmos version's "
                "OmniSetupOverrides does not expose "
                "'parallelism_preset'. "
                "Check inference args implementation."
            )

        return base


# ============================================================
# Internal config
# ============================================================


@dataclass(frozen=True)
class FDConfig:

    seed: int

    guidance: float

    num_steps: int

    fps: int

    action_chunk_size: int

    raw_action_dim: int

    max_action_dim: int

    domain_name: str

    image_size: int


# ============================================================
# Service
# ============================================================


class ForwardDynamicsAVService:

    def __init__(
        self,
        args: ForwardDynamicsServerArgs,
    ) -> None:

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is required."
            )

        # ----------------------------------------------------
        # distributed init
        # ----------------------------------------------------

        maybe_init_distributed()

        rank = _rank()

        log.info(
            f"[fd-av rank={rank}] "
            f"distributed ready "
            f"world_size={_world_size()} "
            f"device={torch.cuda.current_device()}"
        )

        # ----------------------------------------------------
        # setup
        # ----------------------------------------------------

        setup_overrides = (
            args.build_setup_overrides()
        )

        setup_args = (
            setup_overrides.build_setup()
        )

        init_output_dir(
            setup_args.output_dir
        )

        setup_args = (
            disable_runtime_ema_for_frozen_config(
                setup_args
            )
        )

        log.info(
            f"[fd-av rank={rank}] "
            f"loading model "
            f"checkpoint="
            f"{setup_args.checkpoint_path}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # all 8 ranks execute this
        # ----------------------------------------------------

        pipe = OmniInference.create(
            setup_args
        )

        self.pipe = pipe

        self.model = pipe.model

        self.model.eval()

        assert isinstance(
            pipe.setup_args,
            OmniSetupArgs,
        )

        self.setup_args = pipe.setup_args

        # ----------------------------------------------------
        # model action dimension
        # ----------------------------------------------------

        model_max_action_dim = getattr(
            self.model.config,
            "max_action_dim",
            None,
        )

        if not isinstance(
            model_max_action_dim,
            int,
        ):

            model_max_action_dim = 64

        self.cfg = FDConfig(

            seed=int(args.seed),

            guidance=float(
                args.guidance
            ),

            num_steps=int(
                args.num_steps
            ),

            fps=int(args.fps),

            action_chunk_size=int(
                args.action_chunk_size
            ),

            raw_action_dim=int(
                args.raw_action_dim
            ),

            max_action_dim=int(
                model_max_action_dim
            ),

            domain_name=str(
                args.domain_name
            ),

            image_size=int(
                args.image_size
            ),
        )

        # Prevent multiple HTTP threads from
        # simultaneously entering distributed collectives.
        self._request_lock = (
            threading.Lock()
        )

        self._prompt_json_formatter = (
            ActionPromptJsonFormatter(
                caption_key="ai_caption"
            )
            if args.format_prompt_as_json
            else None
        )

        log.info(
            f"[fd-av rank={rank}] READY "
            f"chunk={self.cfg.action_chunk_size} "
            f"raw_dim={self.cfg.raw_action_dim} "
            f"max_dim={self.cfg.max_action_dim}"
        )

    # ========================================================
    # model input video key
    # ========================================================

    def _input_video_key(self) -> str:

        key = getattr(
            self.model,
            "input_video_key",
            None,
        )

        if key is None:

            key = (
                self.model.config
                .input_video_key
            )

        return key

    # ========================================================
    # Prepare request
    # ========================================================

    def _prepare_request(
        self,
        req: dict[str, Any],
    ) -> dict[str, Any]:

        # ----------------------------------------------------
        # image
        # ----------------------------------------------------

        image_b64 = req.get(
            "image"
        )

        if not isinstance(
            image_b64,
            str,
        ):
            raise ValueError(
                "'image' must be a base64 PNG string"
            )

        img = _decode_base64_png(
            image_b64
        )

        # ----------------------------------------------------
        # prompt
        # ----------------------------------------------------

        prompt = req.get(
            "prompt",
            _DEFAULT_PROMPT,
        )

        if not isinstance(
            prompt,
            str,
        ):
            raise ValueError(
                "'prompt' must be string"
            )

        # ----------------------------------------------------
        # domain
        # ----------------------------------------------------

        domain_name = req.get(
            "domain_name",
            self.cfg.domain_name,
        )

        # ----------------------------------------------------
        # action
        # ----------------------------------------------------

        action_raw = np.asarray(
            req.get("action"),
            dtype=np.float32,
        )

        expected = (
            self.cfg.action_chunk_size,
            self.cfg.raw_action_dim,
        )

        if action_raw.shape != expected:

            raise ValueError(
                f"Expected action shape "
                f"{expected}, "
                f"got {action_raw.shape}"
            )

        action_raw = torch.from_numpy(
            action_raw
        )

        # ----------------------------------------------------
        # resize image
        # ----------------------------------------------------

        _, H, W = img.shape

        image_size = int(
            req.get(
                "image_size",
                self.cfg.image_size,
            )
        )

        if H != image_size:

            scale = (
                image_size / H
            )

            new_w = int(
                round(W * scale)
            )

            hwc = (
                img
                .permute(1, 2, 0)
                .numpy()
            )

            resized = Image.fromarray(
                hwc
            ).resize(
                (
                    new_w,
                    image_size,
                ),
                resample=(
                    Image.Resampling.BILINEAR
                ),
            )

            arr = np.asarray(
                resized,
                dtype=np.uint8,
            ).copy()

            img = (
                torch
                .from_numpy(arr)
                .permute(2, 0, 1)
                .contiguous()
            )

        _, final_h, final_w = (
            img.shape
        )

        # ----------------------------------------------------
        # video
        #
        # Same pattern as official policy server:
        # construct full sequence and let sequence_plan
        # determine conditioning.
        # ----------------------------------------------------

        t_frames = (
            self.cfg.action_chunk_size
            + 1
        )

        video = (
            img
            .unsqueeze(1)
            .repeat(
                1,
                t_frames,
                1,
                1,
            )
        )

        # ----------------------------------------------------
        # reflection padding
        # ----------------------------------------------------

        resolution = (
            get_vision_data_resolution(
                (
                    final_h,
                    final_w,
                )
            )
        )

        target_w, target_h = (
            find_closest_target_size(
                final_h,
                final_w,
                resolution,
            )
        )

        pad_dict = {
            "video": video
        }

        reflection_pad_to_target(
            pad_dict,
            ["video"],
            True,
            target_w,
            target_h,
        )

        # ----------------------------------------------------
        # FD sequence plan
        #
        # THIS IS THE CORE CHANGE:
        #
        # wam
        #   ->
        # forward_dynamics
        # ----------------------------------------------------

        sequence_plan = (
            build_sequence_plan_from_mode(
                mode="forward_dynamics",
                video_length=t_frames,
                action_length=(
                    self.cfg.action_chunk_size
                ),
                has_text=True,
            )
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        if (
            self._prompt_json_formatter
            is None
        ):

            augmented_prompt = (
                _augment_prompt(
                    prompt,
                    t_frames=t_frames,
                    fps=self.cfg.fps,
                    height=final_h,
                    width=final_w,
                )
            )

        else:

            formatter_data = {

                "ai_caption":
                    prompt,

                "viewpoint":
                    "ego_view",

                "video":
                    pad_dict["video"],

                "image_size":
                    pad_dict[
                        "image_size"
                    ],

                "conditioning_fps":
                    torch.tensor(
                        self.cfg.fps,
                        dtype=torch.long,
                    ),

                "mode":
                    "forward_dynamics",

                "action":
                    action_raw,

                "idle_frames":
                    torch.tensor(
                        0,
                        dtype=torch.long,
                    ),
            }

            formatted = (
                self
                ._prompt_json_formatter(
                    formatter_data
                )["ai_caption"]
            )

            augmented_prompt = (
                json.dumps(formatted)
                if isinstance(
                    formatted,
                    dict,
                )
                else str(formatted)
            )

        return {

            "video_padded":
                pad_dict["video"],

            "padded_image_size":
                pad_dict["image_size"],

            "action_raw":
                action_raw,

            "prompt":
                augmented_prompt,

            "sequence_plan":
                sequence_plan,

            "domain_name":
                domain_name,
        }

    # ========================================================
    # Build batch
    # ========================================================
    # from cosmos_framework.data.generator.action.action_processing import (
    #     resolve_action_normalization,
    # )

    # normalizer = resolve_action_normalization(
    #     "auto",
    #     norm_stats,
    # )
    def _build_batch(
        self,
        prep: dict[str, Any],
    ) -> dict[str, Any]:

        # ----------------------------------------------------
        # Pad 9D action -> model max_action_dim
        # ----------------------------------------------------

        action = torch.zeros(
            (
                self.cfg.action_chunk_size,
                self.cfg.max_action_dim,
            ),
            dtype=torch.float32,
        )

        action[
            :,
            : self.cfg.raw_action_dim,
        ] = prep["action_raw"]

        input_video_key = (
            self._input_video_key()
        )

        device = _local_device()

        batch = {

            # Official server layout:
            #
            # video:
            # [
            #   [Tensor]
            # ]
            input_video_key: [
                [
                    prep[
                        "video_padded"
                    ]
                ]
            ],

            # Important action processing metadata
            **make_batched_action_processing_fields(
                ActionProcessingRecord(
                    raw_action_dim=(
                        self.cfg.raw_action_dim,
                    ),
                    action_normalizer=None,
                ),
                batch_size=1,
            ),

            "action": [
                [action]
            ],

            "mode": [
                "forward_dynamics"
            ],

            "ai_caption": [
                prep["prompt"]
            ],

            "prompt": [
                prep["prompt"]
            ],

            "conditioning_fps": [
                torch.tensor(
                    self.cfg.fps,
                    dtype=torch.long,
                )
            ],

            "image_size":
                prep[
                    "padded_image_size"
                ]
                .unsqueeze(0)
                .to(device),

            "domain_id": [
                torch.tensor(
                    get_domain_id(
                        prep[
                            "domain_name"
                        ]
                    ),
                    dtype=torch.long,
                )
            ],

            "sequence_plan": [
                prep[
                    "sequence_plan"
                ]
            ],
        }

        return batch

    # ========================================================
    # Local inference
    #
    # ALL ranks execute this.
    # ========================================================

    def _infer_local(self, req: dict[str, Any]) -> dict[str, Any] | None:
        rank = _rank()
        output_mode = req.get("output_mode", "video")
        if output_mode not in {"video", "latent", "both"}:
            raise ValueError(f"Unsupported output_mode={output_mode!r}")

        latent_output_path = req.get("latent_output_path")
        if output_mode in {"latent", "both"} and not isinstance(latent_output_path, str):
            raise ValueError("'latent_output_path' is required for output_mode='latent' or 'both'")

        prep = self._prepare_request(req)
        batch = self._build_batch(prep)
        log.info(f"[fd-av rank={rank}] inference start mode={output_mode}")

        pred_video = None
        latent = None

        with torch.inference_mode():
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            samples = self.model.generate_samples_from_batch(
                batch,
                guidance=self.cfg.guidance,
                seed=[self.cfg.seed],
                num_steps=self.cfg.num_steps,
                has_negative_prompt=False,
            )

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if rank == 0:
                print(f"generate_samples_from_batch: {t1 - t0:.3f}s", flush=True)

            latent = samples["vision"][0]

            if rank == 0:
                print(f"latent: shape={tuple(latent.shape)} dtype={latent.dtype}", flush=True)

            # Save latent only on rank0. No tensor is sent through HTTP.
            if output_mode in {"latent", "both"} and rank == 0:
                path = Path(latent_output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                latent_cpu = latent.detach().cpu()
                torch.save(latent_cpu, path)
                print(f"saved latent: {path}", flush=True)

            # All ranks decode when video output is requested, preserving distributed behavior.
            if output_mode in {"video", "both"}:
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                pred_video = self.model.decode(latent).squeeze(0)

                torch.cuda.synchronize()
                t1 = time.perf_counter()

                if rank == 0:
                    print(f"decode: {t1 - t0:.3f}s", flush=True)

                pred_video = remove_reflection_padding(pred_video, prep["padded_image_size"])

        log.info(f"[fd-av rank={rank}] inference finished mode={output_mode}")

        if rank != 0:
            return None

        result = {"fps": self.cfg.fps, "seed": self.cfg.seed, "output_mode": output_mode}

        if output_mode in {"latent", "both"}:
            result.update({
                "latent_path": str(latent_output_path),
                "latent_shape": list(latent.shape),
                "latent_dtype": str(latent.dtype),
            })

        if output_mode in {"video", "both"}:
            assert pred_video is not None
            video_b64 = _encode_video_frames(pred_video)
            result.update({"video": video_b64, "num_frames": len(video_b64)})

        return result

    # ========================================================
    # Rank0 public inference API
    # ========================================================

    def predict(
        self,
        req: dict[str, Any],
    ) -> dict[str, Any]:

        if _rank() != 0:

            raise RuntimeError(
                "predict() may only be "
                "called on rank0"
            )

        # Extremely important:
        # one distributed collective request
        # at a time.
        with self._request_lock:

            # ----------------------------------------------
            # Broadcast command + request
            # ----------------------------------------------

            if _dist_enabled():

                payload = [
                    {
                        "cmd": "infer",
                        "request": req,
                    }
                ]

                dist.broadcast_object_list(
                    payload,
                    src=0,
                )

            # ----------------------------------------------
            # rank0 also participates
            # ----------------------------------------------

            result = (
                self._infer_local(
                    req
                )
            )

            assert result is not None

            return result

    # ========================================================
    # Worker ranks 1..7
    # ========================================================

    def worker_loop(
        self,
    ) -> None:

        rank = _rank()

        if rank == 0:

            raise RuntimeError(
                "rank0 must not enter "
                "worker_loop()"
            )

        log.info(
            f"[fd-av rank={rank}] "
            f"waiting for rank0 requests"
        )

        while True:

            payload = [None]

            # ----------------------------------------------
            # Block here until rank0 receives HTTP request
            # ----------------------------------------------

            dist.broadcast_object_list(
                payload,
                src=0,
            )

            msg = payload[0]

            if not isinstance(
                msg,
                dict,
            ):

                raise RuntimeError(
                    f"Invalid broadcast "
                    f"message: {msg}"
                )

            cmd = msg.get(
                "cmd"
            )

            if cmd == "shutdown":

                log.info(
                    f"[fd-av rank={rank}] "
                    f"shutdown"
                )

                break

            if cmd != "infer":

                raise RuntimeError(
                    f"Unknown command "
                    f"{cmd!r}"
                )

            req = msg[
                "request"
            ]

            # rank1..7 participate
            self._infer_local(
                req
            )


# ============================================================
# HTTP handler
# ============================================================


class _FDHandler(
    BaseHTTPRequestHandler
):

    server: ThreadingHTTPServer

    def _send_json(
        self,
        code: int,
        payload: dict[str, Any],
    ) -> None:

        data = json.dumps(
            payload
        ).encode("utf-8")

        self.send_response(code)

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(data)),
        )

        self.end_headers()

        self.wfile.write(
            data
        )

    def do_GET(self) -> None:

        if self.path == "/":

            self._send_json(
                200,
                {
                    "status": "ok",
                    "rank": 0,
                    "world_size":
                        _world_size(),
                },
            )

        elif self.path == "/info":

            service = getattr(
                self.server,
                "service",
            )

            self._send_json(
                200,
                {
                    "world_size":
                        _world_size(),

                    "mode":
                        "forward_dynamics",

                    "domain":
                        service.cfg.domain_name,

                    "fps":
                        service.cfg.fps,

                    "action_chunk_size":
                        service
                        .cfg
                        .action_chunk_size,

                    "raw_action_dim":
                        service
                        .cfg
                        .raw_action_dim,

                    "max_action_dim":
                        service
                        .cfg
                        .max_action_dim,
                },
            )

        else:

            self._send_json(
                404,
                {
                    "error":
                        "Not found"
                },
            )

    def do_POST(self) -> None:

        if self.path != "/predict":

            self._send_json(
                404,
                {
                    "error":
                        "Not found"
                },
            )

            return

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            body = (
                self.rfile
                .read(length)
            )

            req = json.loads(
                body.decode("utf-8")
            )

            service = getattr(
                self.server,
                "service",
            )

            result = (
                service.predict(
                    req
                )
            )

            self._send_json(
                200,
                result,
            )

        except Exception as exc:

            traceback.print_exc()

            self._send_json(
                500,
                {
                    "error":
                        str(exc)
                },
            )

    def log_message(
        self,
        format: str,
        *args: Any,
    ) -> None:

        return


# ============================================================
# serve
# ============================================================


def serve(
    args: ForwardDynamicsServerArgs,
) -> None:

    # --------------------------------------------------------
    # ALL ranks load model
    # --------------------------------------------------------

    service = (
        ForwardDynamicsAVService(
            args
        )
    )

    rank = _rank()

    world_size = _world_size()

    log.info(
        f"[fd-av] "
        f"rank={rank}/{world_size} "
        f"model initialized"
    )

    # Ensure every rank has finished model load.
    if _dist_enabled():

        dist.barrier()

    # --------------------------------------------------------
    # rank1..7:
    #
    # no HTTP server
    # --------------------------------------------------------

    if rank != 0:

        service.worker_loop()

        return

    # --------------------------------------------------------
    # rank0 only:
    #
    # HTTP server
    # --------------------------------------------------------

    local_ip = get_local_ip()

    log.info(
        "[fd-av rank0] "
        f"starting HTTP server "
        f"http://{local_ip}:"
        f"{args.port}"
    )

    log.info(
        "[fd-av rank0] "
        f"world_size={world_size}"
    )

    httpd = ThreadingHTTPServer(
        (
            args.host,
            int(args.port),
        ),
        _FDHandler,
    )

    setattr(
        httpd,
        "service",
        service,
    )

    try:

        httpd.serve_forever()

    finally:

        # best effort shutdown command

        if _dist_enabled():

            try:

                payload = [
                    {
                        "cmd":
                            "shutdown"
                    }
                ]

                dist.broadcast_object_list(
                    payload,
                    src=0,
                )

            except Exception:

                pass


# ============================================================
# main
# ============================================================


def main() -> None:

    args = tyro_cli(
        ForwardDynamicsServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )

    serve(args)


if __name__ == "__main__":
    main()