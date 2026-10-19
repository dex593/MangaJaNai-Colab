from dataclasses import replace
from pathlib import Path, PurePosixPath

from manga_colab.model_selector import load_workflow
from manga_colab.scanner import build_plan, output_names, scan_files


def test_recursive_unicode_natural_and_exclude_output(config, write_image):
    config = replace(config, output=config.input / "out")
    for name in ("Chương 01/ページ10.png", "Chương 01/ページ2.jpg", "a/b.bmp", "out/skip.png"):
        write_image(config.input / name)
    files = scan_files(config)
    assert [p.name for p in files] == ["b.bmp", "ページ2.jpg", "ページ10.png"]


def test_duplicates_and_case_collision():
    paths = [PurePosixPath("a/page.jpg"), PurePosixPath("a/page.png"), PurePosixPath("a/Page.PNG")]
    result = output_names(paths, "png")
    assert len(set(value.casefold() for value in result.values())) == 3
    assert all("--" in value for value in result.values())
    assert output_names(paths, "png") == result


def test_content_not_extension_and_corruption(config, write_image):
    write_image(config.input / "gray.png")
    write_image(config.input / "color.png", color=True)
    (config.input / "broken.jpg").write_bytes(b"invalid image")
    items = {item.source: item for item in build_plan(config, load_workflow())}
    assert items["gray.png"].grayscale
    assert not items["color.png"].grayscale
    assert items["broken.jpg"].error


def test_output_names_are_resolved_as_a_whole():
    first = output_names([Path("a.jpg"), Path("a.png")], "png")
    assert first[Path("a.jpg")] != first[Path("a.png")]
