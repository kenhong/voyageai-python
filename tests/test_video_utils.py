import base64
from os import PathLike
from typing import IO, Dict, Optional, Union, Any

import io
import os
from pathlib import Path

import pytest

from voyageai.video_utils import Video, optimize_video


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

    @pytest.mark.parametrize(
        "input_value",
        [
            b"raw-video-bytes",
            bytearray(b"raw-video-bytearray"),
        ],
    )
    def test_optimize_video_from_bytes_like(self, input_value: Union[bytes, bytearray]) -> None:
        video = optimize_video(input_value, resize=False, downsample_fps=False)
        assert isinstance(video, Video)
        assert video.optimized is True
        assert video.to_bytes() == bytes(input_value)

    def test_optimize_video_from_path(self, tmp_path: Path) -> None:
        video_bytes = b"path-video-bytes"
        video_path = tmp_path / "video.bin"
        video_path.write_bytes(video_bytes)

        video = optimize_video(video_path)

        assert isinstance(video, Video)
        assert video.optimized is True
        assert video.to_bytes() == video_bytes

    def test_optimize_video_from_video_instance(self) -> None:
        original = Video(data=b"already-video", optimized=False)
        optimized = optimize_video(original)

        assert isinstance(optimized, Video)
        assert optimized.optimized is True
        assert optimized.to_bytes() == original.to_bytes()


