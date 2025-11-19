from __future__ import annotations

import os
from os import PathLike
from typing import IO, Dict, Optional, Union, Any


class Video:
    """
    Represents a video payload, optionally pre-optimized, ready to be passed
    into Client.multimodal_embed in list-of-list format.

    Internally, this class stores the video bytes and minimal metadata.
    It does NOT decode frames in Python.
    """

    def __init__(
        self,
        data: bytes,
        *,
        optimized: bool = False,
        mime_type: Optional[str] = None,
    ) -> None:
        self._data = data
        self.optimized = optimized
        self.mime_type = mime_type

    @classmethod
    def from_path(
        cls,
        path: Union[str, PathLike[str]],
        *,
        optimize: bool = True,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Video":
        """
        Create a Video object from a local filesystem path.

        If optimize=True, call voyageai.video_utils.optimize_video(...)
        with this path and optimizer_kwargs, and return the optimized Video.

        If optimize=False, ignore optimizer_kwargs and just load bytes from path.
        """
        if optimize:
            # Delegate to the optimizer, normalizing kwargs to an empty dict if needed.
            return optimize_video(
                path,
                **(optimizer_kwargs or {}),
            )

        # optimize is False: read bytes directly and ignore optimizer_kwargs.
        with open(path, "rb") as f:
            data = f.read()
        return cls(data=data, optimized=False)

    @classmethod
    def from_file(
        cls,
        file_obj: IO[bytes],
        *,
        optimize: bool = True,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Video":
        """
        Create a Video object from a file-like object (IO[bytes]).

        If optimize=True, call voyageai.video_utils.optimize_video(...)
        with the raw bytes and optimizer_kwargs, and return the optimized Video.

        If optimize=False, ignore optimizer_kwargs and wrap the bytes directly.
        """
        data = file_obj.read()

        if optimize:
            return optimize_video(
                data,
                **(optimizer_kwargs or {}),
            )

        # optimize is False: wrap the bytes as-is and ignore optimizer_kwargs.
        return cls(data=data, optimized=False)

    def to_bytes(self) -> bytes:
        """
        Return the encoded video as raw bytes.
        """
        return self._data

    def to_file(self, path: Union[str, PathLike[str]]) -> None:
        """
        Save the encoded video bytes to the given path.
        """
        with open(path, "wb") as f:
            f.write(self._data)


def _load_video_bytes(video: Union[str, PathLike[str], bytes, Video]) -> bytes:
    """
    Helper to normalize the supported video input types into raw bytes.
    """
    if isinstance(video, Video):
        return video.to_bytes()

    if isinstance(video, (str, os.PathLike)):
        with open(video, "rb") as f:
            return f.read()

    if isinstance(video, (bytes, bytearray)):
        return bytes(video)

    raise TypeError(
        f"Unsupported video type {type(video)!r}. Expected str, PathLike, bytes, or Video."
    )


def optimize_video(
    video: Union[str, PathLike[str], bytes, Video],
    *,
    resize: bool = True,
    resize_multiple: int = 28,
    downsample_fps: bool = True,
    max_video_tokens: int = 32000,
) -> Video:
    """
    Optimize video for embedding.

    - If `video` is str or PathLike: treat it as a local path.
    - If `video` is bytes: treat it as raw encoded video bytes.
    - If `video` is a Video: re-optimize that video.

    resize:
        If True, resize video dimensions to the nearest multiple of
        `resize_multiple` to improve backend preprocessor performance.

    resize_multiple:
        The integer multiple for width/height rounding (e.g., 28).

    downsample_fps:
        If True, downsample the frame rate to keep within max_video_tokens.

    max_video_tokens:
        Approximate token budget for video frames. The exact mapping
        to frames is backend-defined, but this function is responsible
        for downsampling towards that budget.

    Returns:
        A Video instance containing the optimized video bytes.
    """

    # NOTE: For now, this function is a structured placeholder.
    # TODO: Implement actual video optimization logic (resize, fps downsampling,
    #       frame selection based on `max_video_tokens`, etc.) once the
    #       backend contracts and preferred toolchain (e.g., ffmpeg) are
    #       finalized.

    raw_bytes = _load_video_bytes(video)

    # Placeholder behavior: return the bytes as-is, but mark them as optimized.
    # This keeps the API surface stable while deferring heavy processing.
    return Video(data=raw_bytes, optimized=True, mime_type="video/mp4")


