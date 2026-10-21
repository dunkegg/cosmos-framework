from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wzj_tools.reward_model.latent_dataset import RankingLatentDataset
from wzj_tools.reward_model.reward_model import build_reward_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="temporal",
        choices=["mean", "temporal"],
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
        "--num-workers",
        type=int,
        default=4,
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
        "--max-groups",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return checkpoint


@torch.no_grad()
def evaluate(model, loader, device):
    total = 0

    pair_correct = {
        "gt_mild": 0,
        "gt_medium": 0,
        "gt_bad": 0,
        "mild_medium": 0,
        "mild_bad": 0,
        "medium_bad": 0,
    }

    strict_ranking_correct = 0
    gt_best_correct = 0

    reward_sum = {
        "gt": 0.0,
        "mild": 0.0,
        "medium": 0.0,
        "bad": 0.0,
    }

    reward_sq_sum = {
        "gt": 0.0,
        "mild": 0.0,
        "medium": 0.0,
        "bad": 0.0,
    }

    margin_sum = {
        "gt_mild": 0.0,
        "gt_medium": 0.0,
        "gt_bad": 0.0,
        "mild_medium": 0.0,
        "mild_bad": 0.0,
        "medium_bad": 0.0,
    }

    per_sample = []

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
            dtype=torch.bfloat16,
        ):
            r_gt = model(gt)
            r_mild = model(mild)
            r_medium = model(medium)
            r_bad = model(bad)

        r_gt = r_gt.float()
        r_mild = r_mild.float()
        r_medium = r_medium.float()
        r_bad = r_bad.float()

        batch_size = r_gt.shape[0]
        total += batch_size

        comparisons = {
            "gt_mild": r_gt - r_mild,
            "gt_medium": r_gt - r_medium,
            "gt_bad": r_gt - r_bad,
            "mild_medium": r_mild - r_medium,
            "mild_bad": r_mild - r_bad,
            "medium_bad": r_medium - r_bad,
        }

        for name, margin in comparisons.items():
            pair_correct[name] += (
                margin > 0
            ).sum().item()

            margin_sum[name] += (
                margin.sum().item()
            )

        strict_ranking = (
            (r_gt > r_mild)
            & (r_mild > r_medium)
            & (r_medium > r_bad)
        )

        strict_ranking_correct += (
            strict_ranking.sum().item()
        )

        rewards = torch.stack(
            [
                r_gt,
                r_mild,
                r_medium,
                r_bad,
            ],
            dim=1,
        )

        gt_best = (
            rewards.argmax(dim=1) == 0
        )

        gt_best_correct += (
            gt_best.sum().item()
        )

        reward_dict = {
            "gt": r_gt,
            "mild": r_mild,
            "medium": r_medium,
            "bad": r_bad,
        }

        for name, values in reward_dict.items():
            reward_sum[name] += (
                values.sum().item()
            )

            reward_sq_sum[name] += (
                (values ** 2).sum().item()
            )

        group_ids = batch["group_id"]

        episodes = batch["episode"]

        for i in range(batch_size):
            per_sample.append({
                "group_id":
                    group_ids[i],

                "episode":
                    int(episodes[i]),

                "reward_gt":
                    float(r_gt[i].item()),

                "reward_mild":
                    float(r_mild[i].item()),

                "reward_medium":
                    float(r_medium[i].item()),

                "reward_bad":
                    float(r_bad[i].item()),

                "gt_gt_mild":
                    bool(
                        r_gt[i] >
                        r_mild[i]
                    ),

                "gt_gt_medium":
                    bool(
                        r_gt[i] >
                        r_medium[i]
                    ),

                "gt_gt_bad":
                    bool(
                        r_gt[i] >
                        r_bad[i]
                    ),

                "mild_gt_medium":
                    bool(
                        r_mild[i] >
                        r_medium[i]
                    ),

                "mild_gt_bad":
                    bool(
                        r_mild[i] >
                        r_bad[i]
                    ),

                "medium_gt_bad":
                    bool(
                        r_medium[i] >
                        r_bad[i]
                    ),

                "strict_ranking":
                    bool(
                        strict_ranking[i]
                    ),

                "gt_best":
                    bool(
                        gt_best[i]
                    ),
            })

    if total == 0:
        raise RuntimeError(
            "No validation samples found."
        )

    accuracy = {
        name:
            pair_correct[name] / total
        for name in pair_correct
    }

    margins = {
        name:
            margin_sum[name] / total
        for name in margin_sum
    }

    reward_stats = {}

    for name in reward_sum:
        mean = (
            reward_sum[name] / total
        )

        mean_sq = (
            reward_sq_sum[name] / total
        )

        variance = max(
            mean_sq - mean * mean,
            0.0,
        )

        reward_stats[name] = {
            "mean": mean,
            "std": variance ** 0.5,
        }

    supervised_pair_acc = (
        pair_correct["gt_mild"]
        + pair_correct["gt_medium"]
        + pair_correct["gt_bad"]
    ) / (3 * total)

    results = {
        "num_groups":
            total,

        "supervised_pair_accuracy":
            supervised_pair_acc,

        "pair_accuracy":
            accuracy,

        "mean_margin":
            margins,

        "strict_ranking_accuracy":
            strict_ranking_correct / total,

        "gt_best_accuracy":
            gt_best_correct / total,

        "reward_stats":
            reward_stats,
    }

    return results, per_sample


def print_results(results):
    n = results["num_groups"]
    acc = results["pair_accuracy"]
    margin = results["mean_margin"]
    stats = results["reward_stats"]

    print("")
    print("=" * 72)
    print(f"Evaluation groups: {n}")
    print("=" * 72)

    print("")
    print("[Supervised pairs]")

    print(
        f"GT > mild   : "
        f"{acc['gt_mild'] * 100:.2f}% "
        f"margin={margin['gt_mild']:.4f}"
    )

    print(
        f"GT > medium : "
        f"{acc['gt_medium'] * 100:.2f}% "
        f"margin={margin['gt_medium']:.4f}"
    )

    print(
        f"GT > bad    : "
        f"{acc['gt_bad'] * 100:.2f}% "
        f"margin={margin['gt_bad']:.4f}"
    )

    print(
        f"Overall     : "
        f"{results['supervised_pair_accuracy'] * 100:.2f}%"
    )

    print("")
    print("[Unsupervised ordering]")

    print(
        f"mild > medium : "
        f"{acc['mild_medium'] * 100:.2f}% "
        f"margin={margin['mild_medium']:.4f}"
    )

    print(
        f"mild > bad    : "
        f"{acc['mild_bad'] * 100:.2f}% "
        f"margin={margin['mild_bad']:.4f}"
    )

    print(
        f"medium > bad  : "
        f"{acc['medium_bad'] * 100:.2f}% "
        f"margin={margin['medium_bad']:.4f}"
    )

    print("")
    print("[Full ranking]")

    print(
        f"GT > mild > medium > bad : "
        f"{results['strict_ranking_accuracy'] * 100:.2f}%"
    )

    print(
        f"GT is highest reward      : "
        f"{results['gt_best_accuracy'] * 100:.2f}%"
    )

    print("")
    print("[Reward distribution]")

    for name in [
        "gt",
        "mild",
        "medium",
        "bad",
    ]:
        print(
            f"{name:6s}: "
            f"mean={stats[name]['mean']:.4f} "
            f"std={stats[name]['std']:.4f}"
        )

    print("")
    print("=" * 72)


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for evaluation."
        )

    dataset = RankingLatentDataset(
        manifest_path=args.manifest,
        split="val",
        max_groups=args.max_groups,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_reward_model(
        model_type=args.model,
        latent_dim=48,
        hidden_dim=args.hidden_dim,
    )

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    print(
        f"[checkpoint] "
        f"path={args.checkpoint} "
        f"epoch={checkpoint.get('epoch', 'unknown')}"
    )

    results, per_sample = evaluate(
        model,
        loader,
        device,
    )

    print_results(results)

    if args.output is not None:
        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w"
        ) as f:
            json.dump(
                {
                    "summary": results,
                    "samples": per_sample,
                },
                f,
                indent=2,
            )

        print(
            f"[saved] {output_path}"
        )


if __name__ == "__main__":
    main()