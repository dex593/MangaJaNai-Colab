# Nghiên cứu và tích hợp upstream

Ngày kiểm tra: 2026-10-19. Source: [MangaJaNaiConverterGui](https://github.com/the-database/MangaJaNaiConverterGui/tree/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222), HEAD `e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222`, latest release 1.5.1. Model repo: [MangaJaNai](https://github.com/the-database/MangaJaNai).

## Những file đã kiểm tra

- `backend/src/run_upscale.py`: detector, `get_chain_for_image`, `should_chain_activate_for_image`, ICC reader, preprocessing queue, auto levels, AI call, `final_target_resize`, `save_image`, CLI JSON initialization.
- `backend/src/README.md`: CLI Linux, thư mục models, cấu hình và ICC resources.
- `ViewModels/MainWindowViewModel.cs`: `DEFAULT_WORKFLOW`, GUI GPU selection, model release download sources.
- `backend/resources/default_cli_configuration.json` và `appstate2.json`: CLI config còn color V1; không đồng nhất với GUI V3 hiện tại.
- `backend/src/pyproject.toml`: upstream pins torch 2.9.1/torchvision 0.24.1 nhưng README còn nhắc CUDA 12.1. Không cài nguyên package upstream vào Colab để tránh thay torch runtime.
- `packages/chaiNNer_pytorch/pytorch/io/load_model.py`: Spandrel ModelLoader, EXTRA_REGISTRY, requires_grad=False/eval, precision policy.
- `spandrel_custom`: custom FDAT ArchSupport, cần `install()` cả khi import script bằng module.
- `accelerator_detection.py`, `packages/chaiNNer_pytorch/settings.py`: full device list CPU-first và filtered accelerator list.
- `nodes/impl/pytorch/auto_split.py`, `nodes/impl/upscale/auto_split.py`, tiler, blending, VRAM estimator.

## Tích hợp

Notebook gọi CLI của project. CLI lập plan bằng các hàm detector/chain đã trích nguyên văn, đọc ảnh bằng cùng libvips ICC reader. `backend.py` import script đã pin với một settings JSON hợp lệ, sau đó gọi đồng bộ `preprocess_worker_image`, `ai_upscale_image`, `convert_image_to_grayscale`, `save_image`. Không dùng các thread/process runners của upstream: exception trong thread có thể khiến downstream đợi sentinel và không truyền lỗi chính xác.

Không có backend subprocess trong đường inference: lỗi Python truyền trực tiếp tới CLI; CLI trả nonzero khi có lỗi, notebook hiển thị exit code. `subprocess.run(check=True)` dùng cho git/setup/doctor. Model thật được load bằng upstream `load_model_node`, không có implementation neural inference thay thế.

Phần khởi tạo global của run_upscale.py đọc argv ngay khi import. Adapter cung cấp argv/settings tạm và khôi phục argv sau import; phần `__main__` không chạy. Context mới được tạo mỗi ảnh để các cleanup closures không tích tụ model qua batch.

## Thay đổi được công bố

`scripts/bootstrap_upstream.py` áp dụng hai thay thế có kiểm tra chính xác trên source đã pin:

1. OOM predicate trong `nodes/impl/pytorch/auto_split.py`: bỏ điều kiện coi mọi exception có chữ `CUDA` hoặc `allocate` là OOM; chỉ nhận `torch.cuda.OutOfMemoryError` hoặc thông báo `out of memory`. Lỗi CUDA khác được truyền ra.
2. Nhánh pause gọi nhầm `safe_cuda_cache_empty()` chưa định nghĩa; thay bằng `safe_accelerator_cache_empty(device)` đã được import.

Adapter xử lý thêm, không đổi weights/inference math:

- Sửa mapping context: JSON giữ GUI index CPU-first, context dùng filtered accelerator index và legacy CUDA index đúng; assert thiết bị được chọn.
- Hook callback của splitter để đếm tile thật. Khi splitter trả `Split`, adapter thoát khỏi split và retry toàn ảnh với tile giảm. Tránh nhánh upstream mid-row restart có `start_y = (y * tile_size_x) // new_tile_size_y`, dùng chiều x thay vì y.
- Không cho phép `model is None` hay preprocessing/output NaN/Inf trở thành success.
- `--detector corrected` là opt-in, không thay hành vi mặc định.

Giữ nguyên màu theo implementation upstream, kể cả các chỗ dùng tên RGB/BGR chưa nhất quán. Không tự sửa pipeline màu hoặc thuật toán screentone. Tile ảnh hưởng context của mạng nên không thể đảm bảo output bit-identical giữa các tile size.

## Provenance model

Hai ZIP official đã được tải thật và đọc mọi member dùng trong workflow. `data/models.json` chứa asset id, tag, URL trả từ API, size, upstream digest nếu có, SHA-256 archive đã kiểm chứng, và filename/member path/size/CRC/SHA-256 mỗi model. Runtime lấy URL qua API và đối chiếu metadata, không nối URL filename để đoán.

V3 archive SHA-256 official: `6c26828ad8e142c9e9c7181451257f3f56a0f5c7f897d9c7878ce898404f53e5`.

V1 archive SHA-256 do project tính (upstream API digest=null): `5156f4167875bba51a8ed52bd1c794b0d7277f7103f99b397518066e4dda7e55`.

## Colab runtime

[Colab runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html) và [backend-info](https://github.com/googlecolab/backend-info) được kiểm tra trực tiếp. Snapshot main tại thời điểm nghiên cứu có Python 3.13 wheels của torch 2.11.0, torchvision 0.26.0, numpy 2.1.3 và Pillow 11.3.0. Snapshot CPU không đồng nghĩa GPU runtime của người dùng có cùng CUDA build; script luôn kiểm tra runtime thực tế.

`setup_colab.sh` tạo venv overlay `--system-site-packages`, đặt constraints từ torch/torchvision/torchaudio đang cài. Không chạy lệnh cài PyTorch/CUDA khác. Backend dependencies pin theo source đã nghiên cứu; numpy/OpenCV/Pillow dùng giới hạn tương thích trong `pyproject.toml`. Doctor kiểm tra import, codecs, ICC, FDAT và GPU sau khi cài.

**Native blocker trên Python 3.13:** PyPI `chainner_ext==0.3.10` chỉ có wheel đến CPython 3.12; source hiện tại `chaiNNer-rs` dùng PyO3/numpy Rust 0.20 và release matrix dừng ở 3.12. Không ép load wheel sai ABI, không tự nâng bindings/đổi thuật toán resize. Project giới hạn Python 3.11–3.12 và hướng dẫn chọn Colab runtime **2026.07**, hiện được FAQ công bố dùng Python 3.12.13 và torch 2.11.0. Đây là giới hạn tương thích thực tế, không tuyên bố hỗ trợ default Python 3.13.

Khi nâng upstream: kiểm tra lại GUI workflow, pure functions, device mapping, patch points, model assets/hash; chạy unit tests và smoke test với model thật, rồi mới thay commit. Không tự theo upstream main trong runtime.
