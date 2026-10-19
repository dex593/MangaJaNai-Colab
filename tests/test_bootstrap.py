import subprocess

from scripts import bootstrap_upstream


def test_fresh_clone_can_pin_older_commit_and_reapply_patch(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    subprocess.run(["git", "init", str(remote)], check=True, capture_output=True)
    relative = "MangaJaNaiConverterGui/backend/src"
    source = remote / relative
    (source / "nodes/impl/pytorch").mkdir(parents=True)
    (source / "run_upscale.py").write_text("# pinned source\n")
    patch_path = source / "nodes/impl/pytorch/auto_split.py"
    patch_path.write_text(
        "\n".join(
            before for before, _ in bootstrap_upstream.PATCHES["nodes/impl/pytorch/auto_split.py"]
        )
    )

    def commit(message):
        subprocess.run(["git", "-C", str(remote), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(remote),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                message,
            ],
            check=True,
            capture_output=True,
        )

    commit("pinned")
    pinned = subprocess.check_output(
        ["git", "-C", str(remote), "rev-parse", "HEAD"], text=True
    ).strip()
    (source / "run_upscale.py").write_text("# newer source\n")
    commit("newer")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(bootstrap_upstream, "ROOT", project)
    monkeypatch.setattr(bootstrap_upstream, "UPSTREAM_REPO", str(remote))
    monkeypatch.setattr(bootstrap_upstream, "UPSTREAM_COMMIT", pinned)
    bootstrap_upstream.main()
    bootstrap_upstream.main()
    checkout = project / ".upstream/MangaJaNaiConverterGui" / relative
    assert (checkout / "run_upscale.py").read_text() == "# pinned source\n"
    assert (
        "isinstance(e, torch.cuda.OutOfMemoryError)"
        in (checkout / "nodes/impl/pytorch/auto_split.py").read_text()
    )
