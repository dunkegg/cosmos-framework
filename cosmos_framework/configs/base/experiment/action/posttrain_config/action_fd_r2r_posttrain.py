# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_fd_r2r_posttrain`` — R2R navigation forward-dynamics post-training."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_r2r_lerobot_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader

cs = ConfigStore.instance()


def _make_model_config() -> dict:
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)

    cfg["sound_gen"] = False
    cfg["sound_dim"] = 64
    cfg["sound_latent_fps"] = 25
    cfg["max_num_tokens_after_packing"] = -1
    cfg["resolution"] = "480"
    cfg["activation_checkpointing"]["mode"] = "selective"

    cfg["tokenizer"]["encode_exact_durations"] = [61]

    cfg["diffusion_expert_config"].update(
        base_fps=24,
        enable_fps_modulation=True,
        load_weights_from_pretrained=False,
        patch_spatial=2,
        unified_3d_mrope_temporal_modality_margin=15000,
        unified_3d_mrope_reset_spatial_ids=True,
    )
    cfg["rectified_flow_training_config"].update(
        image_loss_scale=None,
        loss_scale=10.0,
        shift={"256": 3, "480": 5, "720": 10},
        sound_loss_scale=2.0,
        train_time_video_distribution="waver",
        train_time_weight="uniform",
        use_discrete_rf=False,
    )
    cfg["vlm_config"]["tokenizer"]["pretrained_model_name"] = \
        "/mnt/nas/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    
    return cfg


action_fd_r2r_posttrain = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /checkpoint": "gcp"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "training_stats"]},
            {"override /ema": "power"},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_fd",
            group="action_sft",
            name="${now:%Y-%m-%d_%H-%M-%S}_action_fd_r2r_posttrain",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_make_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[
                # "moe_gen",
                # "time_embedder",
                # "vae2llm",
                # "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-04,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            # optimizer_type="AdamW",
            weight_decay=0.05,
        ),
        scheduler=dict(
            cycle_lengths=[20000],
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            lr_scheduler_type="LambdaLinear",
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=50,
            max_iter=20000,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=100, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            straggler_detection=dict(enabled=False, report_freq=50),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                # device_monitor=dict(every_n=200, log_memory_detail=True, save_s3=False, step_size=1),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=50, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=200, gc_level=1, warm_up=1),
                norm_monitor=dict(every_n=100),
                param_count=dict(save_s3=False),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                compile_tokenizer=dict(enabled=False, warmup_resolutions=["480"]),
            ),
        ),
        checkpoint=dict(
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema. (EMA warm-starts from net, see dcp.py)
            keys_to_skip_loading=[
                "net_ema.",
            ],
            load_ema_to_reg=False,
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=250,
            strict_resume=True,
            verbose=True,
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_r2r",
            max_samples_per_batch=1,
            # max_sequence_length="${model.config.max_num_tokens_after_packing}",
            max_sequence_length=None,
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            sound_latent_fps="${model.config.sound_latent_fps}",
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=3,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=2,
                sampler=None,
                datasets=dict(
                    r2r=dict(
                        ratio=1,
                        dataset=L(get_action_r2r_lerobot_sft_dataset)(
                            root="${oc.env:DATASET_PATH}",
                            fps=10.0,
                            chunk_length=60,
                            split="train",
                            mode="forward_dynamics",
                            viewpoint="ego_view",
                            action_normalization=None,
                            use_image_augmentation=False,

                            resolution="480",
                            max_action_dim="${model.config.max_action_dim}",
                            tokenizer_config="${model.config.vlm_config.tokenizer}",

                            cfg_dropout_rate=0.1,
                            append_viewpoint_info=True,
                            append_duration_fps_timestamps=True,
                            append_resolution_info=True,
                            append_idle_frames=False,
                            idle_frames_dropout=0.05,
                            format_prompt_as_json=True,

                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


cs.store(
    group="experiment",
    package="_global_",
    name="action_fd_r2r_posttrain",
    node=action_fd_r2r_posttrain,
)