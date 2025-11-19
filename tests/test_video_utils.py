from os import PathLike
from typing import IO, Dict, Union, Any

import io
from pathlib import Path

import pytest

from voyageai.video_utils import (
    Video,
    optimize_video,
    _compute_target_fps,
)


class TestVideoUtils:
    def test_video_from_path_without_optimize(self, tmp_path: Path) -> None:
        video_bytes = b"fake-video-bytes-from-path"
        video_path = tmp_path / "fake_video.bin"
        video_path.write_bytes(video_bytes)

        video = Video.from_path(video_path, optimize=False, optimizer_kwargs={"ignored": True})

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
            resize: bool = True,
            resize_multiple: int = 28,
            downsample_fps: bool = True,
            max_video_tokens: int = 32000,
        ) -> Video:
            called["video"] = video
            called["resize"] = resize
            called["resize_multiple"] = resize_multiple
            called["downsample_fps"] = downsample_fps
            called["max_video_tokens"] = max_video_tokens
            return Video(data=b"optimized", optimized=True)

        monkeypatch.setattr("voyageai.video_utils.optimize_video", fake_optimize_video)

        video = Video.from_path(
            video_path,
            optimize=True,
            optimizer_kwargs={"resize": False, "max_video_tokens": 12345},
        )

        assert isinstance(video, Video)
        assert video.optimized is True
        assert video.to_bytes() == b"optimized"
        assert called["video"] == video_path
        assert called["resize"] is False
        assert called["resize_multiple"] == 28
        assert called["downsample_fps"] is True
        assert called["max_video_tokens"] == 12345

    def test_video_from_file_without_optimize(self) -> None:
        video_bytes = b"fake-video-bytes-from-file"
        buf: IO[bytes] = io.BytesIO(video_bytes)

        video = Video.from_file(buf, optimize=False, optimizer_kwargs={"ignored": True})

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
            resize: bool = True,
            resize_multiple: int = 28,
            downsample_fps: bool = True,
            max_video_tokens: int = 32000,
        ) -> Video:
            called["video"] = video
            return Video(data=b"optimized-file", optimized=True)

        monkeypatch.setattr("voyageai.video_utils.optimize_video", fake_optimize_video)

        video = Video.from_file(buf, optimize=True, optimizer_kwargs={"resize": False})

        assert isinstance(video, Video)
        assert video.optimized is True
        assert video.to_bytes() == b"optimized-file"
        assert called["video"] == video_bytes

    def test_video_to_bytes_and_to_file(self, tmp_path: Path) -> None:
        video_bytes = b"roundtrip-video-bytes"
        video = Video(data=video_bytes, optimized=False)

        assert video.to_bytes() == video_bytes

        out_path = tmp_path / "out_video.bin"
        video.to_file(out_path)

        assert out_path.read_bytes() == video_bytes

    def test_optimize_video_uses_ffmpeg_pipeline_for_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Sanity-check that optimize_video:
        - Probes metadata,
        - Builds an ffmpeg graph,
        - Returns a Video with optimized=True and MP4 mime type.

        This test stubs out the ffmpeg module used by voyageai.video_utils so it
        does not require the actual ffmpeg binary at test time.
        """

        # Arrange a fake input file.
        video_bytes = b"fake-input-video"
        video_path = tmp_path / "video.bin"
        video_path.write_bytes(video_bytes)

        # Stub ffmpeg used inside voyageai.video_utils.
        calls: Dict[str, Any] = {}

        class FakeStream:
            def __init__(self, label: str) -> None:
                self.label = label
                self.filters: Dict[str, Any] = {}

            def filter(self, name: str, *args, **kwargs) -> "FakeStream":
                self.filters[name] = {"args": args, "kwargs": kwargs}
                return self

        class FakeFFmpegModule:
            class Error(Exception):
                def __init__(self, stderr: bytes = b"") -> None:
                    super().__init__("ffmpeg error")
                    self.stderr = stderr

            @staticmethod
            def probe(path: str) -> Dict[str, Any]:
                calls["probe_path"] = path
                return {
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 640,
                            "height": 360,
                            "r_frame_rate": "30/1",
                            "duration": "1.0",
                        }
                    ],
                    "format": {"duration": "1.0"},
                }

            @staticmethod
            def input(path: str) -> FakeStream:
                calls["input_path"] = path
                return FakeStream("input")

            @staticmethod
            def output(stream: FakeStream, target: str, **kwargs: Any) -> FakeStream:
                calls["output_target"] = target
                calls["output_kwargs"] = kwargs
                return stream

            @staticmethod
            def run(stream: FakeStream, capture_stdout: bool, capture_stderr: bool):
                calls["run_capture_stdout"] = capture_stdout
                calls["run_capture_stderr"] = capture_stderr
                return b"optimized-mp4", b""

        monkeypatch.setattr("voyageai.video_utils.ffmpeg", FakeFFmpegModule)

        # Act
        video = optimize_video(str(video_path))

        # Assert
        assert isinstance(video, Video)
        assert video.optimized is True
        assert video.mime_type == "video/mp4"
        assert video.to_bytes() == b"optimized-mp4"

        # ffmpeg plumbing was exercised.
        assert calls["probe_path"] == str(video_path)
        assert calls["input_path"] == str(video_path)
        assert calls["output_target"] == "pipe:"
        assert calls["run_capture_stdout"] is True
        assert calls["run_capture_stderr"] is True

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


