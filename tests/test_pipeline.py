"""Orchestration tests with an explicitly injected test backend, never AI smoke tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from manga_colab import pipeline
from manga_colab.model_manager import ModelManager


@pytest.fixture
def isolated_backend(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "gpu_info",
        lambda *a, **kw: dict(
            name="test fixture",
            cuda="fixture",
            torch="fixture",
            free=2**30,
            total=2**30,
            compute="fixture",
            fp16=False,
            bf16=False,
        ),
    )
    monkeypatch.setattr(ModelManager, "valid", lambda *a: True)
    monkeypatch.setattr(ModelManager, "ensure", lambda *a, **kw: {})

    class TestBackend:
        """Test-only output generator to exercise durability independently of CUDA."""

        fail = False
        calls = 0

        def __init__(self, config, *args):
            self.config = config

        def process(self, source, destination, chain, callback):
            TestBackend.calls += 1
            if TestBackend.fail:
                raise ValueError("injected test failure")
            with Image.open(source) as image:
                height = round(image.height * self.config.scale)
                Image.new(
                    "RGB", (round(image.width * height / image.height), height), "white"
                ).save(destination)
            return {"oom_retries": 0}

        def close(self):
            return None

    monkeypatch.setattr(pipeline, "Backend", TestBackend)
    return TestBackend


def report(config):
    return json.loads((config.output / ".manga-colab/report.json").read_text(encoding="utf-8"))


def test_success_resume_corruption_and_overwrite(config, write_image, isolated_backend):
    write_image(config.input / "Chương 1/ページ.png")
    assert pipeline.run(config) == 0
    assert report(config)["summary"]["successful"] == 1
    assert pipeline.run(config) == 0
    assert isolated_backend.calls == 1
    output = Path(report(config)["images"][0]["output"])
    output.write_bytes(b"broken")
    assert pipeline.run(config) == 2
    assert output.read_bytes() == b"broken"
    assert pipeline.run(replace(config, overwrite=True)) == 0
    assert isolated_backend.calls == 2


def test_changed_config_cannot_skip(config, write_image, isolated_backend):
    write_image(config.input / "one.png")
    assert pipeline.run(config) == 0
    assert pipeline.run(replace(config, scale=3)) == 2
    assert isolated_backend.calls == 1


def test_changed_model_cannot_skip_or_overwrite_previous_result(
    config, write_image, isolated_backend
):
    write_image(config.input / "one.png")
    assert pipeline.run(config) == 0
    old = (config.output / "one.png").read_bytes()
    changed = replace(config, model="manga-ora")
    assert pipeline.run(changed) == 2
    assert isolated_backend.calls == 1
    assert (config.output / "one.png").read_bytes() == old
    assert pipeline.run(replace(changed, output=config.output / "Manga Ora")) == 0
    assert (
        report(replace(changed, output=config.output / "Manga Ora"))["images"][0]["model"]
        == "2x_Manga_Ora.pth"
    )
    assert pipeline.run(replace(changed, output=config.output / "Manga Ora")) == 0
    assert isolated_backend.calls == 2


@pytest.mark.parametrize("staging", [False, True])
def test_new_images_after_run_keep_old_outputs(config, write_image, isolated_backend, staging):
    if staging:
        config = replace(config, stage=config.input.parent / "stage")
    write_image(config.input / "Chương 1/ページ1.png")
    assert pipeline.run(config) == 0
    output = config.output / "Chương 1/ページ1.png"
    before = (output.read_bytes(), output.stat().st_mtime_ns)
    write_image(config.input / "Chương 1/ページ2.jpg")
    write_image(config.input / "Chương 2/ページ1.png", color=True)
    assert pipeline.run(config) == 0
    summary = report(config)["summary"]
    assert summary["total"] == 3
    assert summary["successful"] == 2
    assert summary["skipped"] == 1
    assert isolated_backend.calls == 3
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before
    assert pipeline.run(config) == 0
    assert report(config)["summary"]["skipped"] == 3
    assert isolated_backend.calls == 3


def test_failed_continues_and_retry_failed(config, write_image, isolated_backend):
    write_image(config.input / "one.png")
    write_image(config.input / "two.png")
    isolated_backend.fail = True
    assert pipeline.run(config) == 2
    assert report(config)["summary"]["failed"] == 2
    isolated_backend.fail = False
    assert pipeline.run(config) == 2
    assert report(config)["summary"]["skipped"] == 2
    assert pipeline.run(replace(config, retry_failed=True)) == 0
    assert report(config)["summary"]["successful"] == 2


def test_017_added_after_001_and_005_only_upscales_new_image(config, write_image, isolated_backend):
    for name in ("001", "005"):
        write_image(config.input / f"{name}.jpg")
    assert pipeline.run(config) == 0
    old = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in config.output.glob("*.png")
    }
    write_image(config.input / "017.jpg")
    assert pipeline.run(config) == 0
    summary = report(config)["summary"]
    assert (summary["total"], summary["successful"], summary["skipped"]) == (3, 1, 2)
    assert isolated_backend.calls == 3
    assert (config.output / "017.png").is_file()
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == saved for path, saved in old.items())


def test_stage_and_duplicate_names(config, write_image, isolated_backend):
    write_image(config.input / "one.jpg")
    write_image(config.input / "one.png")
    assert pipeline.run(replace(config, stage=config.input.parent / "stage")) == 0
    outputs = [row["output"] for row in report(config)["images"]]
    assert len(set(outputs)) == 2
    assert not list((config.input.parent / "stage").iterdir())


def test_corrupt_and_valid_files_are_separate(config, write_image, isolated_backend):
    (config.input / "bad.jpg").write_bytes(b"broken")
    write_image(config.input / "good.png")
    assert pipeline.run(config) == 2
    assert report(config)["summary"]["successful"] == 1
    assert report(config)["summary"]["failed"] == 1


def test_dry_run_does_not_use_gpu_or_download(config, write_image, monkeypatch):
    write_image(config.input / "one.png")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run attempted inference/download")

    monkeypatch.setattr(pipeline, "gpu_info", forbidden)
    monkeypatch.setattr(ModelManager, "ensure", forbidden)
    assert pipeline.run(config, dry_run=True) == 0
    assert not (config.output / "one.png").exists()


def test_interrupt_never_commits_partial_output(config, write_image, isolated_backend, monkeypatch):
    write_image(config.input / "one.png")

    def interrupt(self, source, destination, *args):
        destination.write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(isolated_backend, "process", interrupt)
    assert pipeline.run(config) == 130
    assert not (config.output / "one.png").exists()
    assert not list(config.output.glob(".manga-*.png"))
