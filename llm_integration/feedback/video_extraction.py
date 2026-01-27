"""Video frame extraction utilities for visual feedback."""

import base64
import logging
from pathlib import Path

import numpy as np

# Use module-level logger (can't import from logging_config at module load time)
logger = logging.getLogger("llm_integration.video_extraction")

# Try to import cv2 for video frame extraction
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Video frame extraction disabled.")


def extract_key_frames(
    video_path: str | Path, num_frames: int = 4, resize: tuple[int, int] = (320, 240)
) -> list[str]:
    """
    Extract evenly-spaced key frames from a video as base64-encoded images.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        resize: Target size (width, height) for efficiency

    Returns:
        List of base64-encoded PNG images
    """
    if not CV2_AVAILABLE:
        return []

    video_path = Path(video_path)
    if not video_path.exists():
        return []

    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            num_frames = max(1, total_frames)

        # Get evenly spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames_base64 = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize for efficiency
                frame = cv2.resize(frame, resize)
                _, buffer = cv2.imencode(".png", frame)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                frames_base64.append(img_base64)

        cap.release()
        return frames_base64

    except Exception as e:
        logger.warning(f"Failed to extract frames from {video_path}: {e}")
        return []


def create_visual_feedback(
    run_dir: Path, iteration: int, num_frames: int = 4
) -> list[str]:
    """
    Extract key frames from planned and simulated trajectory videos.

    Args:
        run_dir: Directory containing iteration results
        iteration: Iteration number
        num_frames: Number of frames to extract per video

    Returns:
        List of base64-encoded images (planned frames first, then simulated)
    """
    images: list[str] = []

    # Extract from planned trajectory
    planned_video = run_dir / f"planned_traj_iter_{iteration}.mp4"
    if planned_video.exists():
        planned_frames = extract_key_frames(planned_video, num_frames)
        images.extend(planned_frames)

    # Extract from simulation
    sim_video = run_dir / f"simulation_iter_{iteration}.mp4"
    if sim_video.exists():
        sim_frames = extract_key_frames(sim_video, num_frames)
        images.extend(sim_frames)

    return images
