"""Downloader safety and meaningful failure-path tests."""

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.download_medhallu import download_and_save


def test_download_reports_hugging_face_network_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_load_dataset(*args: object, **kwargs: object) -> None:
        raise ConnectionError("offline test network")

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=failing_load_dataset),
    )

    with pytest.raises(RuntimeError, match="pqa_labeled.*ConnectionError"):
        download_and_save(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
        )


def test_download_refuses_existing_outputs_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    existing = raw_root / "pqa_labeled"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: pytest.fail("should not download")),
    )

    with pytest.raises(FileExistsError, match="pass --allow-overwrite"):
        download_and_save(raw_root=raw_root, processed_root=processed_root)

    assert marker.read_text(encoding="utf-8") == "keep"