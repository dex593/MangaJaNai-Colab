import json
from dataclasses import replace

from manga_colab.checkpoint import Checkpoint
from manga_colab.model_selector import load_workflow
from manga_colab.storage import sha256


def test_resume_checks_checksum_and_config(config, write_image):
    output = write_image(config.output / "ページ.png")
    workflow = load_workflow()
    fingerprint = config.fingerprint(workflow, "model1")
    checkpoint = Checkpoint(config.output / "manifest.jsonl")
    record = dict(
        source="ページ.png",
        source_hash="source1",
        fingerprint=fingerprint,
        output=str(output),
        output_hash=sha256(output),
        dimensions=[24, 32],
        status="success",
    )
    checkpoint.append(record)
    checkpoint = Checkpoint(checkpoint.path)
    assert checkpoint.completed("ページ.png", "source1", fingerprint, output)
    assert not checkpoint.completed("ページ.png", "source2", fingerprint, output)
    for changed in (
        replace(config, scale=3),
        replace(config, detector="corrected"),
        replace(config, format="webp"),
    ):
        assert not checkpoint.completed(
            "ページ.png", "source1", changed.fingerprint(workflow, "model1"), output
        )
    assert not checkpoint.completed(
        "ページ.png", "source1", config.fingerprint(workflow, "model2"), output
    )
    workflow["Chains"]["$values"].reverse()
    assert not checkpoint.completed(
        "ページ.png", "source1", config.fingerprint(workflow, "model1"), output
    )
    output.write_bytes(b"corrupt")
    assert not checkpoint.completed("ページ.png", "source1", fingerprint, output)


def test_prepared_recovery_and_torn_tail(config, write_image):
    output = write_image(config.output / "test.png")
    path = config.output / "manifest.jsonl"
    record = dict(
        source="a",
        source_hash="s",
        fingerprint="f",
        output=str(output),
        output_hash=sha256(output),
        dimensions=[24, 32],
        status="prepared",
    )
    path.write_bytes((json.dumps(record) + '\n{"source":').encode())
    checkpoint = Checkpoint(path)
    assert checkpoint.completed("a", "s", "f", output)
    checkpoint.append({**record, "status": "success"})
    assert Checkpoint(path).records["a"]["status"] == "success"
