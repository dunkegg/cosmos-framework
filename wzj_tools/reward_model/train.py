from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from wzj_tools.reward_model.latent_dataset import RankingLatentDataset
from wzj_tools.reward_model.reward_model import build_reward_model


def pairwise_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    return -F.logsigmoid(
        positive - negative
    ).mean()


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    amp_dtype,
):
    model.eval()

    total_groups = 0

    correct_mild = 0
    correct_medium = 0
    correct_bad = 0

    margin_mild = 0.0
    margin_medium = 0.0
    margin_bad = 0.0

    reward_gt = 0.0
    reward_mild = 0.0
    reward_medium = 0.0
    reward_bad = 0.0

    for batch in loader:
        gt = batch["gt"].to(
            device,
            non_blocking=True,
        )

        mild = batch["mild"].to(
            device,
            non_blocking=True,
        )

        medium = batch["medium"].to(
            device,
            non_blocking=True,
        )

        bad = batch["bad"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        ):
            r_gt = model(gt)
            r_mild = model(mild)
            r_medium = model(medium)
            r_bad = model(bad)

        n = gt.shape[0]

        total_groups += n

        correct_mild += (
            r_gt > r_mild
        ).sum().item()

        correct_medium += (
            r_gt > r_medium
        ).sum().item()

        correct_bad += (
            r_gt > r_bad
        ).sum().item()

        margin_mild += (
            r_gt - r_mild
        ).sum().item()

        margin_medium += (
            r_gt - r_medium
        ).sum().item()

        margin_bad += (
            r_gt - r_bad
        ).sum().item()

        reward_gt += r_gt.sum().item()
        reward_mild += r_mild.sum().item()
        reward_medium += r_medium.sum().item()
        reward_bad += r_bad.sum().item()

    metrics = {
        "acc_mild": correct_mild / total_groups,
        "acc_medium": correct_medium / total_groups,
        "acc_bad": correct_bad / total_groups,
        "acc_overall": (
            correct_mild
            + correct_medium
            + correct_bad
        ) / (3 * total_groups),

        "margin_mild": margin_mild / total_groups,
        "margin_medium": margin_medium / total_groups,
        "margin_bad": margin_bad / total_groups,

        "reward_gt": reward_gt / total_groups,
        "reward_mild": reward_mild / total_groups,
        "reward_medium": reward_medium / total_groups,
        "reward_bad": reward_bad / total_groups,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="wzj_train_output/reward_v0",
    )

    parser.add_argument(
        "--model",
        choices=["mean", "temporal"],
        default="temporal",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--mild-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--medium-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--bad-weight",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    torch.manual_seed(args.split_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = RankingLatentDataset(
        manifest_path=args.manifest,
        split="train",
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        max_groups=args.max_groups,
    )

    val_dataset = RankingLatentDataset(
        manifest_path=args.manifest,
        split="val",
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        max_groups=args.max_groups,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "Reward V0 training currently expects CUDA."
        )

    amp_dtype = torch.bfloat16

    model = build_reward_model(
        model_type=args.model,
        latent_dim=48,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"[model] type={args.model} "
        f"params={num_params:,} "
        f"device={device}"
    )

    best_acc = -1.0
    global_step = 0

    weights_sum = (
        args.mild_weight
        + args.medium_weight
        + args.bad_weight
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        running_count = 0

        for batch_idx, batch in enumerate(
            train_loader
        ):
            gt = batch["gt"].to(
                device,
                non_blocking=True,
            )

            mild = batch["mild"].to(
                device,
                non_blocking=True,
            )

            medium = batch["medium"].to(
                device,
                non_blocking=True,
            )

            bad = batch["bad"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
            ):
                r_gt = model(gt)
                r_mild = model(mild)
                r_medium = model(medium)
                r_bad = model(bad)

                loss_mild = pairwise_loss(
                    r_gt,
                    r_mild,
                )

                loss_medium = pairwise_loss(
                    r_gt,
                    r_medium,
                )

                loss_bad = pairwise_loss(
                    r_gt,
                    r_bad,
                )

                loss = (
                    args.mild_weight
                    * loss_mild
                    + args.medium_weight
                    * loss_medium
                    + args.bad_weight
                    * loss_bad
                ) / weights_sum

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            with torch.no_grad():
                acc = (
                    (r_gt > r_mild).float().mean()
                    + (r_gt > r_medium).float().mean()
                    + (r_gt > r_bad).float().mean()
                ) / 3.0

            batch_size = gt.shape[0]

            running_loss += (
                loss.item()
                * batch_size
            )

            running_acc += (
                acc.item()
                * batch_size
            )

            running_count += batch_size
            global_step += 1

            if global_step % 20 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={global_step} "
                    f"loss={loss.item():.6f} "
                    f"acc={acc.item():.4f} "
                    f"r_gt={r_gt.mean().item():.4f} "
                    f"r_mild={r_mild.mean().item():.4f} "
                    f"r_medium={r_medium.mean().item():.4f} "
                    f"r_bad={r_bad.mean().item():.4f}"
                )

        train_loss = (
            running_loss
            / running_count
        )

        train_acc = (
            running_acc
            / running_count
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            amp_dtype,
        )

        print(
            f"[epoch] "
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['acc_overall']:.4f} "
            f"mild={val_metrics['acc_mild']:.4f} "
            f"medium={val_metrics['acc_medium']:.4f} "
            f"bad={val_metrics['acc_bad']:.4f} "
            f"margin_mild={val_metrics['margin_mild']:.4f} "
            f"margin_medium={val_metrics['margin_medium']:.4f} "
            f"margin_bad={val_metrics['margin_bad']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_type": args.model,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        }

        torch.save(
            checkpoint,
            output_dir / "latest.pt",
        )

        if (
            val_metrics["acc_overall"]
            > best_acc
        ):
            best_acc = (
                val_metrics[
                    "acc_overall"
                ]
            )

            torch.save(
                checkpoint,
                output_dir / "best.pt",
            )

            print(
                f"[best] "
                f"epoch={epoch} "
                f"val_acc={best_acc:.4f}"
            )

        with (
            output_dir
            / "metrics.jsonl"
        ).open("a") as f:
            f.write(
                json.dumps({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    **val_metrics,
                })
                + "\n"
            )


if __name__ == "__main__":
    main()