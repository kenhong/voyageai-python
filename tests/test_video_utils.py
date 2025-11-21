from os import PathLike
from typing import IO, Dict, Union, Any

import io
from pathlib import Path

import pytest

from voyageai._base import _get_client_config
from voyageai.video_utils import (
    Video,
    optimize_video,
    _compute_target_fps,
    _parse_fps,
)

try:
    import ffmpeg  # type: ignore[import]
except ImportError:  # pragma: no cover - handled lazily in functions
    ffmpeg = None  # type: ignore[assignment]



class TestVideoUtils:
    def test_video_from_path_without_optimize(self, tmp_path: Path) -> None:
        video_bytes = b"fake-video-bytes-from-path"
        video_path = tmp_path / "fake_video.bin"
        video_path.write_bytes(video_bytes)

        video = Video.from_path(
            video_path,
            model="voyage-multimodal-3.5",
            optimize=False,
            optimizer_kwargs={"ignored": True},
        )

        assert isinstance(video, Video)
        assert video.optimized is False
        assert video.to_bytes() == video_bytes

    def test_video_from_path_with_optimize_calls_optimize_video(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        video_bytes = b"fake-video-bytes-from-path-opt"
        video_path = tmp_path / "fake_video_opt.bin"
        video_path.write_bytes(video_bytes)

        called: Dict[str, Any] = {}

        def fake_optimize_video(
            video: Union[str, PathLike[str], bytes, Video],
            *,
            model: str,
            resize: bool = True,
            resize_multiple: int = 28,
            downsample_fps: bool = True,
            max_video_tokens: int = 32000,
        ) -> Video:
            called["video"] = video
            called["model"] = model
            called["resize"] = resize
            called["resize_multiple"] = resize_multiple
            called["downsample_fps"] = downsample_fps
            called["max_video_tokens"] = max_video_tokens
            return Video(data=b"optimized", model=model, optimized=True)

        monkeypatch.setattr("voyageai.video_utils.optimize_video", fake_optimize_video)

        video = Video.from_path(
            video_path,
            model="voyage-multimodal-3.5",
            optimize=True,
            optimizer_kwargs={"resize": False, "max_video_tokens": 12345},
        )

        assert isinstance(video, Video)
        assert video.optimized is True
        assert video.to_bytes() == b"optimized"
        assert called["video"] == video_path
        assert called["model"] == "voyage-multimodal-3.5"
        assert called["resize"] is False
        assert called["resize_multiple"] == 28
        assert called["downsample_fps"] is True
        assert called["max_video_tokens"] == 12345

    def test_video_from_file_without_optimize(self) -> None:
        video_bytes = b"fake-video-bytes-from-file"
        buf: IO[bytes] = io.BytesIO(video_bytes)

        video = Video.from_file(
            buf,
            model="voyage-multimodal-3.5",
            optimize=False,
            optimizer_kwargs={"ignored": True},
        )

        assert isinstance(video, Video)
        assert video.optimized is False
        assert video.to_bytes() == video_bytes

    def test_video_from_file_with_optimize_calls_optimize_video(self, monkeypatch: pytest.MonkeyPatch) -> None:
        video_bytes = b"fake-video-bytes-from-file-opt"
        buf: IO[bytes] = io.BytesIO(video_bytes)

        called: Dict[str, Any] = {}

        def fake_optimize_video(
            video: Union[str, PathLike[str], bytes, Video],
            *,
            model: str,
            resize: bool = True,
            resize_multiple: int = 28,
            downsample_fps: bool = True,
            max_video_tokens: int = 32000,
        ) -> Video:
            called["video"] = video
            called["model"] = model
            return Video(data=b"optimized-file", model=model, optimized=True)

        monkeypatch.setattr("voyageai.video_utils.optimize_video", fake_optimize_video)

        video = Video.from_file(
            buf,
            model="voyage-multimodal-3.5",
            optimize=True,
            optimizer_kwargs={"resize": False},
        )

        assert isinstance(video, Video)
        assert video.optimized is True
        assert video.to_bytes() == b"optimized-file"
        assert called["video"] == video_bytes
        assert called["model"] == "voyage-multimodal-3.5"

    def test_video_to_bytes_and_to_file(self, tmp_path: Path) -> None:
        video_bytes = b"roundtrip-video-bytes"
        video = Video(data=video_bytes, model="voyage-multimodal-3.5", optimized=False)

        assert video.to_bytes() == video_bytes

        out_path = tmp_path / "out_video.bin"
        video.to_file(out_path)

        assert out_path.read_bytes() == video_bytes


    @pytest.mark.integration
    def test_optimize_video_e2e_example_video(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        End-to-end test for optimize_video using the real ffmpeg-python and
        example_video_01.mp4, validating that:

        - Output bytes form a valid MP4 with a video stream.
        - num_frames, num_pixels, and estimated_num_tokens on Video are
          consistent with the client_config and probed metadata.
        """

        input_path = Path("tests/example_video_01.mp4")
        assert input_path.is_file(), "tests/example_video_01.mp4 must exist for this test"

        video = optimize_video(str(input_path), model="voyage-multimodal-3.5")

        assert isinstance(video, Video)
        assert video.optimized is True
        assert video.mime_type == "video/mp4"
        assert len(video.to_bytes()) > 0

        # Persist optimized bytes to disk and probe with real ffmpeg.
        output_path = tmp_path / "optimized_video.mp4"
        video.to_file(output_path)

        probe = ffmpeg.probe(str(output_path))  # type: ignore[union-attr]
        stream = next(
            s for s in probe["streams"] if s.get("codec_type") == "video"
        )
        width = int(stream["width"])
        height = int(stream["height"])
        duration = float(stream.get("duration", probe.get("format", {}).get("duration", 0.0)))
        fps = _parse_fps(stream.get("r_frame_rate", "0/0"))

        assert width > 0 and height > 0
        assert duration > 0
        assert fps > 0

        # Recompute expected usage using the same client_config and rules.
        cfg = _get_client_config("voyage-multimodal-3.5")
        min_pixels = cfg["multimodal_video_pixels_min"]
        max_pixels = cfg["multimodal_video_pixels_max"]
        ratio = cfg["multimodal_video_to_tokens_ratio"]

        frames = int(fps * duration)
        if frames % 2 == 1:
            frames -= 1
        assert frames > 0

        pixels_per_frame_raw = width * height
        assert pixels_per_frame_raw > 0

        pixels_per_frame = max(
            min_pixels,
            min(max_pixels, pixels_per_frame_raw),
        )

        expected_num_pixels = pixels_per_frame * frames
        expected_tokens = max(
            1,
            (pixels_per_frame // max(ratio, 1)) * frames,
        )

        assert video.num_frames == frames
        assert video.num_pixels == expected_num_pixels
        assert video.estimated_num_tokens == expected_tokens

    @pytest.mark.parametrize(
        "original_fps,duration,max_tokens,tokens_per_frame,expected_relation",
        [
            (30.0, 10.0, 1600, 16, "lt"),  # should downsample
            (30.0, 1.0, 10_000_000, 16, "eq"),  # effectively unchanged
        ],
    )
    def test_compute_target_fps_behavior(
        self,
        original_fps: float,
        duration: float,
        max_tokens: int,
        tokens_per_frame: int,
        expected_relation: str,
    ) -> None:
        target_fps = _compute_target_fps(
            original_fps=original_fps,
            duration_sec=duration,
            max_video_tokens=max_tokens,
            tokens_per_frame=tokens_per_frame,
        )

        if expected_relation == "lt":
            assert target_fps <= original_fps
        elif expected_relation == "eq":
            assert target_fps == original_fps


