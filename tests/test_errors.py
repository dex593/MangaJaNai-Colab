from manga_colab.logger import error_message


def test_model_loader_root_cause_visible():
    try:
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as cause:
            raise ValueError("Model unsupported") from cause
    except ValueError as exc:
        assert "CUDA out of memory" in error_message(exc)
