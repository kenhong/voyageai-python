from __future__ import annotations

import os
import tempfile
from os import PathLike
from typing import IO, Dict, Optional, Union, Any

try:
    import ffmpeg  # type: ignore[import]
except ImportError:  # pragma: no cover - handled lazily in functions
    ffmpeg = None  # type: ignore[assignment]


# Default approximate pixel-to-token ratio for multimodal video when computing
# a target FPS during optimization. This mirrors the image-based client config
# behavior (e.g., 1120 pixels per token) but may be adjusted as backend
# contracts evolve.
_DEFAULT_VIDEO_PIXEL_TO_TOKEN_RATIO: int = 1120


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

        # optimize is False: read bytes directly.
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

        # optimize is False: wrap the bytes as-is.
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

    def estimated_num_tokens(
        self,
        *,
        model: Optional[str] = None,
        client_config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Estimate the number of tokens represented by this video for a given model.

        This uses the multimodal video settings from the model's client config:

        - multimodal_video_pixels_min
        - multimodal_video_pixels_max
        - multimodal_video_to_tokens_ratio

        and applies the formula:

            pixels_per_frame = max(min_pixels, min(max_pixels, W * H))
            frames = floor(fps * duration) rounded down to nearest multiple of 2
            tokens = (pixels_per_frame // pixel_to_token_ratio) * frames

        If client_config is not provided, `model` must be given so that the
        configuration can be loaded via the internal client-config helper.
        """
        if ffmpeg is None:
            raise ImportError(
                "ffmpeg-python is required to estimate video tokens. "
                "Install `ffmpeg-python` and ensure `ffmpeg` is available on PATH."
            )

        if client_config is None:
            if model is None:
                raise ValueError(
                    "Either `client_config` or `model` must be provided to estimate tokens."
                )
            # Lazy import to avoid circular imports at module import time.
            try:
                from voyageai._base import _get_client_config  # type: ignore
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    "Unable to load client config for video token estimation."
                ) from exc

            client_config = _get_client_config(model)

        min_pixels = int(client_config["multimodal_video_pixels_min"])
        max_pixels = int(client_config["multimodal_video_pixels_max"])
        pixel_to_token_ratio = int(client_config["multimodal_video_to_tokens_ratio"])

        temp_file: Optional[tempfile.NamedTemporaryFile] = None
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(self._data)
            temp_file.flush()
            meta = _probe_video(temp_file.name)
            width = meta["width"]
            height = meta["height"]
            duration = meta["duration"]
            fps = _parse_fps(meta["r_frame_rate"])

            if fps <= 0 or duration <= 0:
                return 0

            pixels_per_frame = max(
                min_pixels,
                min(max_pixels, width * height),
            )
            tokens_per_frame = max(pixels_per_frame // max(pixel_to_token_ratio, 1), 1)

            frames = int(fps * duration)
            # Round down to nearest multiple of 2 (drop last frame if odd).
            if frames % 2 == 1:
                frames -= 1
            if frames <= 0:
                return 0

            return tokens_per_frame * frames
        finally:
            if temp_file is not None:
                temp_name = temp_file.name
                try:
                    temp_file.close()
                except Exception:
                    pass
                if temp_name and os.path.exists(temp_name):
                    try:
                        os.unlink(temp_name)
                    except OSError:
                        pass


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


def _ensure_ffmpeg_available() -> None:
    if ffmpeg is None:
        raise ImportError(
            "ffmpeg-python is required for video optimization. "
            "Install `ffmpeg-python` and ensure `ffmpeg` is available on PATH."
        )


def _probe_video(path: Union[str, PathLike[str]]) -> Dict[str, Any]:
    """
    Probe video metadata using ffmpeg.probe and return key properties.
    """
    _ensure_ffmpeg_available()

    probe = ffmpeg.probe(str(path))  # type: ignore[union-attr]
    video_stream = next(
        (s for s in probe["streams"] if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError("No video stream found in input video")

    format_info = probe.get("format", {})
    duration_str = video_stream.get("duration", format_info.get("duration", 0.0))

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "r_frame_rate": video_stream.get("r_frame_rate", "0/0"),
        "duration": float(duration_str),
    }


def _parse_fps(r_frame_rate: str) -> float:
    num, _, den = r_frame_rate.partition("/")
    try:
        num_f = float(num)
        den_f = float(den) if den else 1.0
        return num_f / den_f if den_f != 0 else 0.0
    except ValueError:
        return 0.0


def _round_to_multiple(value: int, multiple: int) -> int:
    """
    Round `value` to the nearest multiple of `multiple`, at least `multiple`.
    """
    if multiple <= 0:
        return value
    rounded = int(round(value / multiple) * multiple)
    return max(multiple, rounded)


def _compute_target_fps(
    original_fps: float,
    duration_sec: float,
    max_video_tokens: int,
    tokens_per_frame: int = 16,
) -> float:
    """
    Approximate a target fps given max_video_tokens.

    Assumes roughly `tokens_per_frame` tokens per frame.
    This is a heuristic and may be tuned later or aligned more closely
    with server-side accounting.
    """
    if original_fps <= 0 or duration_sec <= 0:
        return original_fps

    if max_video_tokens <= 0:
        return original_fps

    max_frames = max_video_tokens // max(tokens_per_frame, 1)
    approx_fps_limit = max_frames / duration_sec
    if approx_fps_limit <= 0:
        return original_fps

    target_fps = min(original_fps, approx_fps_limit)

    # Encourage an even frame count by slightly adjusting fps so that
    # floor(target_fps * duration) is even, approximating "drop last frame if odd".
    frames = int(target_fps * duration_sec)
    if frames > 1 and frames % 2 == 1:
        frames -= 1
        target_fps = frames / duration_sec

    return target_fps


def optimize_video(
    video: Union[str, PathLike[str], bytes, Video],
    *,
    resize: bool = True,
    resize_multiple: int = 28,
    downsample_fps: bool = True,
    max_video_tokens: int = 32000,
) -> Video:
    """
    Optimize video using ffmpeg-python.

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
        Approximate token budget for video frames. This function uses a simple
        heuristic tokens-per-frame estimate to select a target fps.

    Returns:
        A Video instance containing the optimized MP4 bytes.
    """
    _ensure_ffmpeg_available()

    temp_file: Optional[tempfile.NamedTemporaryFile] = None
    input_path: Union[str, PathLike[str]]

    # 1. Normalize input to a file on disk for ffmpeg to read.
    if isinstance(video, Video):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(video.to_bytes())
        temp_file.flush()
        input_path = temp_file.name
    elif isinstance(video, (str, os.PathLike)):
        input_path = str(video)
    elif isinstance(video, (bytes, bytearray)):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(bytes(video))
        temp_file.flush()
        input_path = temp_file.name
    else:
        raise TypeError(f"Unsupported video input type: {type(video)!r}")

    try:
        # 2. Probe metadata and compute target dimensions / fps.
        meta = _probe_video(input_path)
        width = meta["width"]
        height = meta["height"]
        duration = meta["duration"]
        original_fps = _parse_fps(meta["r_frame_rate"])

        if resize:
            target_width = _round_to_multiple(width, resize_multiple)
            target_height = _round_to_multiple(height, resize_multiple)
        else:
            target_width = width
            target_height = height

        if downsample_fps and max_video_tokens is not None:
            # Estimate tokens-per-frame based on the current spatial resolution,
            # using the same pixel-to-token ratio as the multimodal image config.
            # This avoids a hard-coded constant and scales with resolution.
            pixels_per_frame = width * height
            tokens_per_frame = max(
                1,
                pixels_per_frame // max(_DEFAULT_VIDEO_PIXEL_TO_TOKEN_RATIO, 1),
            )
            target_fps = _compute_target_fps(
                original_fps=original_fps,
                duration_sec=duration,
                max_video_tokens=max_video_tokens,
                tokens_per_frame=tokens_per_frame,
            )
        else:
            target_fps = original_fps

        # 3. Build the ffmpeg filter graph.
        stream = ffmpeg.input(str(input_path))  # type: ignore[union-attr]

        if target_fps > 0:
            stream = stream.filter("fps", fps=target_fps)

        if resize:
            stream = stream.filter(
                "scale",
                target_width,
                target_height,
                flags="bicubic",
            )

        # 4. Output settings: short GOP, no audio, H.264, yuv420p, rate control.
        x264_params = (
            "bframes=0:"
            "ref=1:"
            "cabac=0:"
            "weightp=0:"
            "deblock=0,0:"
            "scenecut=0:"
            "keyint=60:"
            "min-keyint=60"
        )

        output_kwargs = {
            "format": "mp4",
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "preset": "veryfast",
            "crf": 20,
            "maxrate": "6M",
            "bufsize": "12M",
            "movflags": "+faststart",
            "an": None,  # drop audio
            "r": target_fps if target_fps > 0 else None,
            "x264_params": x264_params,
        }

        # Remove None values (ffmpeg-python does not accept them as kwargs).
        output_kwargs = {k: v for k, v in output_kwargs.items() if v is not None}

        stream = ffmpeg.output(stream, "pipe:", **output_kwargs)  # type: ignore[union-attr]

        try:
            out, err = ffmpeg.run(
                stream,
                capture_stdout=True,
                capture_stderr=True,
            )  # type: ignore[union-attr]
        except Exception as e:
            # If this is an ffmpeg.Error, surface stderr for easier debugging.
            if hasattr(e, "stderr"):
                stderr = getattr(e, "stderr")
                decoded = (
                    stderr.decode("utf-8", errors="ignore")
                    if isinstance(stderr, (bytes, bytearray))
                    else str(stderr)
                )
                raise RuntimeError(f"ffmpeg optimization failed: {decoded}") from e
            raise

        # Successful optimization; wrap in Video object.
        return Video(data=out, mime_type="video/mp4", optimized=True)
    finally:
        if temp_file is not None:
            temp_name = temp_file.name
            try:
                temp_file.close()
            except Exception:
                pass
            if temp_name and os.path.exists(temp_name):
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass


