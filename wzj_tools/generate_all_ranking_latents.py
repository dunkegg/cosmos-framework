#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from wzj_tools.cosmos_latent_client import cosmos_predict_latent, load_action_json

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', path.name)]


# def read_image_frame(image_root: str | Path, frame_idx: int) -> np.ndarray:
#     image_root = Path(image_root)
#     image_paths = sorted((p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
#     if not image_paths:
#         raise RuntimeError(f'No images found in: {image_root}')
#     if not 0 <= frame_idx < len(image_paths):
#         raise IndexError(f'frame_idx={frame_idx} out of range [0, {len(image_paths) - 1}]')
#     image_path = image_paths[frame_idx]
#     frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
#     if frame is None:
#         raise RuntimeError(f'Failed to read image: {image_path}')
#     return frame
def read_image_frame(image_root: str | Path, episode_idx: int, frame_idx: int) -> np.ndarray:
    image_root = Path(image_root)
    prefix = f"episode_{episode_idx:06d}_"

    image_paths = sorted(
        image_root.glob(f"{prefix}*"),
        key=natural_key,
    )

    if not image_paths:
        raise RuntimeError(f"No images found for episode {episode_idx} in {image_root}")

    if not 0 <= frame_idx < len(image_paths):
        raise IndexError(
            f"episode={episode_idx}, frame_idx={frame_idx}, "
            f"num_images={len(image_paths)}"
        )

    image_path = image_paths[frame_idx]
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    return frame

def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f'Invalid manifest JSONL line {line_no}: {e}') from e
    return rows


def save_manifest(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def get_image_root(row: dict) -> Path:
    source = row['source']
    if 'image_root' in source:
        return Path(source['image_root'])
    video_path = Path(source['video_path'])
    return video_path if video_path.is_dir() else video_path.parent


class LatentChunkWriter:
    def __init__(self, output_dir: Path, chunk_size: int = 128, dtype: str = 'float16', start_chunk_idx: int = 0):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.chunk_idx = start_chunk_idx
        self.latents = []
        self.keys = []

    def _convert_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if self.dtype == 'float16':
            return x.to(torch.float16)
        if self.dtype == 'bfloat16':
            return x.to(torch.bfloat16)
        if self.dtype == 'float32':
            return x.to(torch.float32)
        raise ValueError(f'Unsupported dtype={self.dtype}')

    def add(self, key: str, latent: torch.Tensor) -> dict:
        chunk_idx = self.chunk_idx
        latent_index = len(self.latents)
        latent = self._convert_dtype(latent.detach().cpu())
        self.latents.append(latent)
        self.keys.append(key)
        info = {
            'latent_chunk': str(self.output_dir / f'chunk-{chunk_idx:06d}.pt'),
            'latent_index': latent_index,
            'latent_shape': list(latent.shape),
            'latent_dtype': str(latent.dtype),
        }
        if len(self.latents) >= self.chunk_size:
            self.flush()
        return info

    def flush(self) -> None:
        if not self.latents:
            return
        shapes = {tuple(x.shape) for x in self.latents}
        if len(shapes) != 1:
            raise RuntimeError(f'Latent shapes differ inside chunk: {shapes}')
        path = self.output_dir / f'chunk-{self.chunk_idx:06d}.pt'
        latents = torch.stack(self.latents, dim=0)
        torch.save({'latents': latents, 'keys': self.keys}, path)
        print(f'saved chunk: {path} shape={tuple(latents.shape)} dtype={latents.dtype}')
        self.latents = []
        self.keys = []
        self.chunk_idx += 1

    def close(self) -> None:
        self.flush()


def next_chunk_index(latent_dir: Path) -> int:
    indices = []
    for path in latent_dir.glob('chunk-*.pt'):
        m = re.match(r'chunk-(\d+)\.pt$', path.name)
        if m:
            indices.append(int(m.group(1)))
    return max(indices) + 1 if indices else 0


def already_done(candidate: dict) -> bool:
    chunk = candidate.get('latent_chunk')
    index = candidate.get('latent_index')
    return chunk is not None and index is not None and Path(chunk).exists()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=None)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--latent-dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    p.add_argument('--tmp-dir', type=Path, default=None)
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save-every-groups', type=int, default=1)
    p.add_argument('--max-groups', type=int, default=0)
    args = p.parse_args()

    manifest_path = args.manifest
    output_root = args.output_root or manifest_path.parent
    latent_dir = output_root / 'ranking_latents'
    tmp_dir = args.tmp_dir or output_root / 'tmp_latents'
    latent_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    print('loaded groups:', len(rows))

    start_chunk_idx = next_chunk_index(latent_dir) if args.resume else 0
    writer = LatentChunkWriter(latent_dir, args.chunk_size, args.latent_dtype, start_chunk_idx)
    processed_groups = processed_candidates = skipped_candidates = 0
    print(f'starting from chunk index: {start_chunk_idx}')
    try:
        for group_idx, row in enumerate(rows):
            if args.max_groups > 0 and processed_groups >= args.max_groups:
                break

            group_id = row['group_id']
            frame_idx = int(row['current_frame'])
            episode_idx = int(row["episode_index"])
            image_root = get_image_root(row)
            pending = [c for c in row['candidates'] if not (args.resume and already_done(c))]
            if not pending:
                skipped_candidates += len(row['candidates'])
                continue

            # frame = read_image_frame(image_root, frame_idx)

            frame = read_image_frame(
                image_root,
                episode_idx,
                frame_idx,
            )
            print('\n' + '=' * 80)
            print(f'group {group_idx + 1}/{len(rows)}: {group_id}')
            print('subtask:', row['subtask_text'])
            print('frame:', frame_idx)
            print('image_root:', image_root)
            print('pending:', [c['candidate_id'] for c in pending])

            for candidate in pending:
                candidate_id = candidate['candidate_id']
                action = load_action_json(candidate['action_path'])
                tmp_latent = tmp_dir / f'{group_id}_{candidate_id}.pt'

                print('-' * 80)
                print('candidate:', candidate_id)
                result = cosmos_predict_latent(frame, action, tmp_latent, output_mode='latent')
                print(f"group_id={group_id}, candidate_id={candidate_id}, latent_shape={result.get('latent_shape')}, latent_dtype={result.get('latent_dtype')}, latent_path={result.get('latent_path')}")
                returned_path = Path(result['latent_path'])
                if not returned_path.exists():
                    raise RuntimeError(f'Latent file not found after server response: {returned_path}')

                latent = torch.load(returned_path, map_location='cpu')
                info = writer.add(f'{group_id}/{candidate_id}', latent)
                candidate.update(info)
                candidate['server_latent_dtype'] = result.get('latent_dtype')
                candidate['server_latent_shape'] = result.get('latent_shape')

                returned_path.unlink(missing_ok=True)
                processed_candidates += 1

            processed_groups += 1
            if processed_groups % args.save_every_groups == 0:
                save_manifest(manifest_path, rows)
                print(f'manifest checkpoint saved after {processed_groups} groups')

    finally:
        writer.close()
        save_manifest(manifest_path, rows)

    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    print('\n' + '=' * 80)
    print('DONE')
    print('processed groups:', processed_groups)
    print('processed candidates:', processed_candidates)
    print('skipped candidates:', skipped_candidates)
    print('latent dir:', latent_dir)
    print('manifest:', manifest_path)


if __name__ == '__main__':
    main()
