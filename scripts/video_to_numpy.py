# video_to_numpy.py

import math
import os
from contextlib import suppress
from typing import Literal, Optional, Tuple

import av
import numpy as np
from av.video.frame import VideoFrame
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

DEFAULT_BUFFER_CAPACITY = 256
MAX_INITIAL_BUFFER_FRAMES = 2048


def video_to_numpy(
    path: str,
    *,
    pix_fmt: Literal["rgb24", "bgr24", "gray"] = "rgb24",
    max_frames: Optional[int] = None,
    stride: int = 1,
    resize: Optional[Tuple[int, int]] = None,  # (width, height)
    show_progress: bool = False,
) -> np.ndarray:
    """
    Decode a video into a NumPy array of frames using PyAV (FFmpeg).

    Args:
        path: Path to the input video file.
        pix_fmt: Output pixel format. Common fast options:
                 - "rgb24" (H, W, 3, dtype=uint8)
                 - "bgr24" (H, W, 3, dtype=uint8) for OpenCV compatibility
                 - "gray"  (H, W, 1, dtype=uint8)
        max_frames: If provided, stop after this many frames.
        stride: Keep one frame every `stride` frames (e.g., 2 halves frame rate).
        resize: Optional (width, height) resize using FFmpeg’s scaler. This is fast.
        show_progress: If True, display a Rich progress bar while decoding.

    Returns:
        frames: NumPy array with shape (N, H, W, C) and dtype=uint8.
    """
    # Open the container and select the first video stream
    container = av.open(path)
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        container.close()
        raise ValueError("No video stream found in file: {}".format(path))

    # Tuning for speed: let FFmpeg do multithreaded decoding
    with suppress(AttributeError):
        setattr(stream, "thread_type", "AUTO")
        setattr(stream, "thread_count", os.cpu_count() or 1)

    expected_total: Optional[int] = None
    if max_frames is not None:
        expected_total = max_frames
    elif stream.frames and stream.frames > 0:
        expected_total = math.ceil(stream.frames / stride)

    kept = 0
    decoded = 0
    initial_capacity = expected_total or DEFAULT_BUFFER_CAPACITY
    buffer_capacity = max(1, min(initial_capacity, MAX_INITIAL_BUFFER_FRAMES))
    frames_buffer: Optional[np.ndarray] = None

    progress: Optional[Progress] = None
    task_id: Optional[int] = None

    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(text_format="[magenta]{task.completed}/{task.total} ({task.percentage:>3.0f}%)"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        task_id = progress.add_task("Decoding frames", total=expected_total)

    # Decode packets -> frames
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                if not isinstance(frame, VideoFrame):
                    continue
                decoded += 1

                # Stride filtering
                if (decoded - 1) % stride != 0:
                    continue

                # Convert to desired pixel format (and resize if requested)
                if resize is not None:
                    width, height = resize
                    frame = frame.reformat(width=width, height=height, format=pix_fmt)
                else:
                    frame = frame.reformat(format=pix_fmt)

                # to_ndarray returns a contiguous uint8 array of shape (H, W, C)
                arr = frame.to_ndarray()
                if arr.ndim == 2:
                    # Gray frames come as (H, W); expand to (H, W, 1) for consistency
                    arr = arr[..., None]

                if frames_buffer is None:
                    frame_shape = arr.shape
                    buffer_capacity = max(buffer_capacity, 1)
                    frames_buffer = np.empty((buffer_capacity, *frame_shape), dtype=np.uint8)
                elif kept >= buffer_capacity:
                    new_capacity = buffer_capacity * 2
                    frame_shape = frames_buffer.shape[1:]
                    new_buffer = np.empty((new_capacity, *frame_shape), dtype=np.uint8)
                    new_buffer[:kept] = frames_buffer[:kept]
                    frames_buffer = new_buffer
                    buffer_capacity = new_capacity

                frames_buffer[kept] = arr
                kept += 1
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)

                if max_frames is not None and kept >= max_frames:
                    break
            if max_frames is not None and kept >= max_frames:
                break
    finally:
        if progress is not None:
            progress.stop()

    container.close()

    if frames_buffer is None or kept == 0:
        # Return an empty array with a consistent shape if nothing decoded
        if pix_fmt == "gray":
            channels = 1
        else:
            channels = 3
        return np.empty((0, 0, 0, channels), dtype=np.uint8)

    # Trim unused tail (if we over-allocated) and return contiguous view
    return frames_buffer[:kept]


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_to_numpy.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    frames = video_to_numpy(
        video_path,
        pix_fmt="rgb24",
        max_frames=None,
        stride=1,
        resize=None,
        show_progress=True,
    )
    print(f"Decoded frames: {frames.shape}, dtype: {frames.dtype}")