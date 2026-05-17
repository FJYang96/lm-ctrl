"""Extract key frames from planned and RL tracking videos for visual comparison.

To help with maybe potenial LLM suggestions

"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def trim_static_tail(frames: list[np.ndarray], *, min_tail: int = 10, diff_threshold: float = 0.5) -> list[np.ndarray]:
    """Drop a repeated final hold so comparison frames cover the active motion."""
    if len(frames) <= min_tail:
        return frames
    last = frames[-1]
    changed_idx = None
    for idx in range(len(frames) - 2, -1, -1):
        diff = np.mean(np.abs(frames[idx].astype(np.float32) - last.astype(np.float32)))
        if diff > diff_threshold:
            changed_idx = idx
            break
    if changed_idx is None:
        return frames
    tail_len = len(frames) - changed_idx - 1
    if tail_len < min_tail:
        return frames
    return frames[: changed_idx + 2]


def extract_and_save(video_path: Path, output_dir: Path, num_frames: int) -> int:
    """Extract evenly-spaced frames from a video and save as PNGs.

    Returns number of frames saved.
    """
    frames = trim_static_tail(read_video_frames(video_path))
    total = len(frames)
    if total == 0:
        print(f"  No frames in {video_path}")
        return 0

    n = min(num_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, idx in enumerate(indices):
        frame = frames[int(idx)]
        if cv2.imwrite(str(output_dir / f"frame_{i + 1:03d}.png"), frame):
            saved += 1
    return saved


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            description="Extract frames from planned and RL tracking videos"
        )
        parser.add_argument("--planned", type=str, default="results/planned_traj.mp4")
        parser.add_argument("--rl", type=str, default="results/rl_tracking.mp4")
        parser.add_argument("--output-dir", type=str, default="results/comparison")
        parser.add_argument("--num-frames", type=int, default=20)
        args = parser.parse_args()

    planned_path = Path(args.planned)
    rl_path = Path(args.rl)
    output_dir = Path(args.output_dir)

    if not planned_path.exists():
        print(f"Planned video not found: {planned_path}")
        return
    if not rl_path.exists():
        print(f"RL video not found: {rl_path}")
        return

    planned_dir = output_dir / "planned_trajectory_frames"
    rl_dir = output_dir / "rl_trajectory_frames"

    print(f"Extracting {args.num_frames} frames from each video...")

    n_planned = extract_and_save(planned_path, planned_dir, args.num_frames)
    print(f"  Planned: {n_planned} frames -> {planned_dir}")

    n_rl = extract_and_save(rl_path, rl_dir, args.num_frames)
    print(f"  RL:      {n_rl} frames -> {rl_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from planned and RL tracking videos"
    )
    parser.add_argument(
        "--planned",
        type=str,
        default="results/planned_traj.mp4",
        help="Path to planned trajectory video",
    )
    parser.add_argument(
        "--rl",
        type=str,
        default="results/rl_tracking.mp4",
        help="Path to RL tracking video",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Directory to save frame folders",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of frames to extract per video",
    )
    args = parser.parse_args()
    main(args)
