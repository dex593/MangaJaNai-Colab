# Kết quả xác minh

Ngày chạy gần nhất: 2026-10-20. Không dùng kết quả giả lập làm bằng chứng AI inference.

## Unit/integration ở cấp điều phối

- Windows: Python 3.12.6, NumPy 2.1.2, OpenCV 4.12.0, Pillow 10.4.0, Rich 13.9.2, libvips 8.16.1.
- Linux/WSL: Python 3.12.3, môi trường riêng cài từ `pyproject.toml` + pyvips 3.0.0/pyvips-binary 8.16.1.
- `python -m pytest -q`: **221 passed** trên Linux; **219 passed, 2 skipped** trên Windows (bản 0.3.1). Hai bài kiểm tra gửi tín hiệu dừng cho cả nhóm tiến trình chỉ chạy trên Linux (môi trường Colab). Test descriptor GFISRV2 cần Torch/Spandrel; môi trường test tối thiểu không cài backend sẽ skip riêng bài này.
- `ruff check manga_colab tests scripts`: pass.
- `bash -n scripts/setup_colab.sh scripts/check_environment.sh`: pass trên Linux.
- Notebook: sáu code cells parse được bằng Python AST, không có secrets/token, không có inference engine trong notebook.

Phạm vi test: grayscale/color, pure B/W/uint8 overflow, threshold, mọi ranh giới height, scale 2/4 và chain priority, missing model, HTTP Range retries/truncated responses, invalid cache checksum, failed download không publish, manifest truncated tail, prepared recovery, source/config/model/workflow thay đổi, output corrupt, Unicode, duplicate filenames, recursive/excluded folders, multi-GPU index mapping, settings JSON, staging, dry-run, retry-failed, Ctrl+C và OOM control flow.

`tests/test_pipeline.py` dùng test double ghi ảnh để cô lập logic checkpoint/file handling. `tests/test_oom.py` mô phỏng tín hiệu OOM, không làm cạn VRAM thật. Các test này không chứng minh chất lượng model.

## Log và hướng dẫn phiên bản 0.1.1

- Kiểm tra bằng subprocess thật: stdout, stderr, tiếng Việt/Nhật xuất hiện ở tiến trình gọi; sự kiện hiển thị được nhận **trước khi** tiến trình con kết thúc; lỗi cài đặt lưu log đầy đủ và hiện phần cuối lỗi.
- Linux: ngắt tiến trình gửi SIGINT và tiếp tục đọc thông báo checkpoint trước khi nhận mã thoát 130.
- Kiểm tra HTML thoát ký tự trong tên file, không chứa mã điều khiển ANSI; các lần cập nhật tiến độ dùng cùng display ID. Kiểm tra phần nối IPython bằng test double, không coi đó là một phiên Colab thật.
- Kiểm tra thêm ảnh sau lượt đầu, có và không dùng chỗ lưu tạm: chỉ ảnh mới chạy, nội dung và thời điểm sửa output cũ giữ nguyên.
- Chạy lại bằng **GPU và hai model 2x thật** nêu dưới: lượt đầu 2 thành công; thêm một ảnh rồi chạy lại được **1 thành công, 2 bỏ qua**, output cũ giữ nguyên cả byte và mtime. Toàn bộ lần chạy đi qua kênh truyền log mới.
- Render bảng log HTML của lần chạy thật trong Chromium và xem ảnh chụp để kiểm tra bố cục, tiếng Việt, độ dễ đọc. Chưa kiểm tra trực tiếp trong giao diện Colab.
- Kiểm tra notebook có sáu cell, các ô dùng tên dễ hiểu và ánh xạ đúng sang tham số CLI; Bước 5 tạo lại tham số mỗi lần chạy và gọi hàm truyền log.

## Biểu mẫu và nhận ảnh mới, phiên bản 0.1.2

- Biểu mẫu còn 10 ô, tên tiếng Việt có dấu. Bỏ các ô địa chỉ chương trình, nhánh, mã phiên bản và thiết lập kỹ thuật. Kiểm tra cả sáu cell bằng Python AST và thực thi cell cấu hình để xác minh các giá trị mặc định.
- Bước 5 gọi hàm làm mới Drive trong kernel trước khi chạy CLI. Dùng API công khai `flush_and_unmount` rồi `mount`; đối chiếu [source Google Colab](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py). Nếu lưu dữ liệu thất bại, không gọi ép kết nối lại; nếu kết nối thất bại hoặc không thấy thư mục, không chạy Upscale. Các nhánh này kiểm tra bằng test double, **chưa chứng minh Drive thật hết trễ cập nhật**.
- Kiểm tra Cell 5 tạo lại tham số và gọi làm mới mỗi lượt; đường dẫn ngoài Drive không làm mới kết nối. Kiểm tra log quét có tổng số ảnh và không gắn tên ảnh cuối.
- Tái lập `001.jpg`, `005.jpg`, thêm `017.jpg` bằng test backend: lượt sau **3 ảnh, 1 thành công, 2 bỏ qua**; byte và thời gian sửa hai kết quả cũ không đổi. Đây là kiểm tra quét/resume trên filesystem cục bộ, không phải thử tải ảnh qua trang Google Drive.
- Không chạy lại GPU inference cho thay đổi giao diện/kết nối này. Kết quả inference thật trước đó được giữ bên dưới; thuật toán AI không thay đổi.

## Lựa chọn xử lý ảnh cũ, phiên bản 0.1.3

- Gộp hai checkbox trái ngược thành một danh sách chọn: **Chỉ Upscale ảnh chưa làm** hoặc **Upscale lại tất cả ảnh**. Mặc định giữ kết quả cũ.
- Biểu mẫu còn 9 ô với nhãn ngắn không có dấu gạch dưới; giữ hướng dẫn tiếng Việt đầy đủ ngay phía trên mỗi ô.
- Thực thi Cell 1 và Cell 5 khi đổi qua lại giữa hai lựa chọn trong cùng phiên: kiểm tra cả giá trị cấu hình và tham số CLI, không có trường hợp bật đồng thời resume và overwrite. Giá trị ngoài hai lựa chọn bị từ chối. Các bài kiểm tra này dùng test double cho CLI, không chạy AI hay kết nối Drive.
- Chưa kiểm tra trực tiếp cách hiển thị trong giao diện Colab. Không chạy lại GPU inference vì thay đổi chỉ nằm ở lựa chọn trong notebook.

## Smoke test inference thật

Thiết bị: **NVIDIA GeForce GTX 1050 Ti 4 GB**, compute 6.1, Windows, PyTorch **2.5.1+cu121**, torchvision 0.20.1+cu121, CUDA 12.1, **FP32**. Backend pin `e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222`; Spandrel 0.4.1 + upstream FDAT.

Ảnh input hợp lệ được vẽ cục bộ để không đưa manga có bản quyền vào repository. Một ảnh grayscale 64×80 có chữ và pattern chấm; một ảnh RGB 64×64. Model tải thật qua GitHub Range downloader, kiểm tra SHA-256. Đây là smoke test nhỏ, không phải benchmark throughput trang manga thật.

| Thử nghiệm | Model chạy thật | Output đã đọc và hash | Kết quả |
|---|---|---|---|
| Target 2x grayscale, tile 64 | `2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth` | 128×160 PNG | PASS |
| Target 2x color, tile 64 | `2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.safetensors` | 128×128 PNG | PASS |
| Target 1.5x grayscale, auto tile + staging | cùng model 2x, ICC final downscale gốc | 96×120 PNG | PASS |
| Target 1.5x color, auto tile + staging | cùng model 2x, Lanczos final downscale gốc | 96×96 PNG | PASS |
| Target 3x grayscale, tile 64 | `4x_MangaJaNai_1200p_V1_ESRGAN_70k.pth`, ICC downscale | 192×240 PNG | PASS |
| Target 3x color, tile 64 | `4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.safetensors`, Lanczos downscale | 192×192 PNG | PASS |
| Chạy lại batch 2x cùng cấu hình | Kiểm tra output + manifest | 2 skipped, không chạy inference lại | PASS |

`doctor --check-models` đã xác minh load các model 2x thực bằng Spandrel. Hai official archives đã được tải đầy đủ khi tạo catalog, kiểm tra ZIP CRC, size và digest nếu upstream cung cấp. Runtime selective download đã chạy thật cho bốn model trên; không tải model không cần thiết.

## Tương thích PyTorch mới trên Linux

Đã cài **PyTorch 2.11.0+cpu / torchvision 0.26.0+cpu** trong môi trường Linux Python 3.12.3 riêng. Spandrel 0.4.1 và custom FDAT của upstream load hai model 2x chính thức thành công; chạy tensor 16×16 qua từng model cho output 32×32 hữu hạn. `chainner_ext 0.3.10` chạy CubicCatrom resize 32×32 → 24×24 thành công. Đây là kiểm tra API/native trên Linux với PyTorch cùng phiên bản runtime Colab 2026.07, **không phải kiểm thử CUDA/Drive trên Colab**.

## Eula và Manga Ora, phiên bản 0.2.0

Ngày 2026-10-20: tải hai tệp thật từ nguồn ghi trong `extra_models.json`, sau đó tải lại qua chính `ModelManager` vào cache riêng để kiểm chứng đường tải của ứng dụng. Eula từ Oracle Object Storage do OpenModelDB liên kết; Ora từ link Google Drive người dùng cung cấp. Cả hai khớp dung lượng và SHA-256 công bố trong metadata OpenModelDB đã pin. `gdown 6.4.2` tải được mà không dùng cookie đăng nhập hay token.

Chạy backend gốc trên **GTX 1050 Ti 4 GB, FP32, PyTorch 2.5.1+cu121**:

| Model | Target | Kết quả đã decode và hash |
|---|---|---|
| Eula ESRGAN 4x, 1 kênh | 4x | 64×80 → 256×320 PNG |
| Eula ESRGAN 4x, 1 kênh | 2x | 64×80 → 128×160 PNG, qua final downscale ICC upstream |
| Manga Ora DAT 2x, RGB | 2x | Grayscale 64×80 → 128×160; RGB 64×64 → 128×128 PNG |
| Manga Ora DAT 2x, RGB | 1.5x | Grayscale → 96×120; RGB → 96×96 PNG, qua final downscale upstream |

`doctor --check-models` xác minh hai model load được, đúng kiến trúc, số kênh và hệ số Upscale. Trong Linux **Python 3.12.3 / PyTorch 2.11.0+cpu / Spandrel 0.4.1**, cả hai chạy tensor 16×16 thành công: Eula cho tensor 1 kênh 64×64, Ora cho tensor RGB 32×32, tất cả giá trị hữu hạn. Chưa xác minh FP16 hay GPU Colab cho hai model này.

Chạy thêm từng model trên GPU thật với staging: lượt đầu **1 thành công**, lượt chạy lại **1 bỏ qua, 0 thành công mới**. Nội dung và thời điểm sửa tệp kết quả giữ nguyên. Đây là kiểm tra trên ổ đĩa cục bộ, chưa kiểm tra mount Google Drive thật.

Các bài test mới kiểm tra lựa chọn model theo CLI/notebook, giới hạn Ora 2x, Eula từ chối ảnh màu, RGB đen/trắng tuyệt đối, workflow/settings JSON, checksum/progress, gián đoạn/quota/sai dung lượng, dọn `.part`, cache hit, thay model không skip/ghi đè sai, thư mục kết quả riêng và loại kết quả model khác khỏi quét input. Test download lỗi dùng dữ liệu giả để kiểm tra xử lý lỗi, tách biệt với tải thật và inference thật ở trên.

Tái lập inference:

```bash
python scripts/smoke_test.py --model eula-digimanga --scale 4
python scripts/smoke_test.py --model eula-digimanga --scale 2
python scripts/smoke_test.py --model manga-ora --scale 2
python scripts/smoke_test.py --model manga-ora --scale 1.5
```

## Bộ nhớ và tốc độ Ora, phiên bản 0.2.1

- Kiểm tra lựa chọn precision với capability T4/L4/A100, FP32 bắt buộc, tile theo VRAM trống, cache bộ nhớ tái sử dụng, giới hạn tile sau OOM theo từng model và cấu hình tile thủ công. Các capability này được giả lập trong unit tests, không phải các GPU đã truy cập thật.
- Đo Ora thật trên GTX 1050 Ti / PyTorch 2.5.1+cu121 / FP32. Hai ảnh tự tạo 800×1200, output 1600×2400. Giới hạn allocator PyTorch còn 2 GiB để tạo OOM thật mà tránh Windows chuyển dữ liệu sang RAM.
- Bản 0.2.0: **51.37 giây, 4 lần OOM/retry**, peak tensor GPU **1602.76 MiB**. Bản 0.2.1: **50.15 giây, 0 lần OOM/retry**, peak tensor GPU **1144.43 MiB**; hai ảnh đều dùng tile 320. Đây là một lượt đo cho mỗi bản, mức chênh thời gian nhỏ, không chứng minh tăng tốc đáng kể hoặc tốc độ trên T4.
- Các phép forward riêng FP32 đã chạy tile 128/256/320/384/512. Tile 512 đạt peak tensor khoảng 2629 MiB trên máy này. Các ngưỡng tile là khởi đầu thận trọng, vẫn có OOM fallback.
- Chạy BF16 có autocast trên GTX 1050 Ti thành công với tile 128, nhưng đây là đường mô phỏng và không được dùng để khẳng định hiệu suất BF16 trên T4/L4/A100. Chính sách mới không chọn đường mô phỏng cho Ora.

## MangaScale GFISRV2, phiên bản 0.3.0

- Liệt kê folder Drive thật, xác định đúng bản `.safetensors`, tải thật và tính SHA-256. Tải lại vào cache riêng qua `ModelManager`; hash và size khớp. Không tải bản ONNX, không thêm model vào Git.
- Giữ nguyên source GFISRV2 từ resselt; unit test kiểm tra hash source, descriptor cho kết quả bằng forward gốc sau clamp, từ chối metadata biến thể khác và weights thiếu key. Không coi trọng số ngẫu nhiên của unit test là bằng chứng chất lượng AI.
- Model thật có **4.208.736 parameters**, 3 kênh vào/ra, scale 4. Trên GTX 1050 Ti, forward FP32 và FP16 + autocast cho input 31×47, 64×80, 512×512 đều đúng kích thước 4x và hữu hạn. Với 512×512, peak tensor GPU lần lượt khoảng **755 MiB / 564 MiB**; đây không phải VRAM tổng của tiến trình.
- Pipeline upstream chạy ảnh màu và grayscale thật ở target **4x** và **2x**, output được decode/hash; 2x dùng final downscale gốc. `doctor --check-models` kiểm tra thêm load qua registry Spandrel.
- Hai trang tự tạo **800×1200 → 1600×2400**, FP32, auto tile, staging, cùng GTX 1050 Ti và allocator giới hạn 2 GiB: **26.56 giây, 0 OOM/retry, peak tensor 522.82 MiB**. Ora trong lượt đo 0.2.1 ở trên mất 50.15 giây với cùng hai input và target; đây là một mẫu nhỏ trên GPU khác T4, có khác biệt staging và output, không phải đánh giá chất lượng hay cam kết mức tăng tốc Colab.
- Chạy lại hai trang: **2 skipped**, nội dung và mtime output giữ nguyên. Linux **PyTorch 2.11.0+cpu** load weights thật và chạy input 17×23 → 68×92 hữu hạn.
- Chưa chạy GFISRV2 trên T4/L4/A100, chưa kiểm chứng batch dài hoặc so sánh chất lượng manga thật. Không dùng số FPS trong bài giới thiệu tác giả làm kết quả đo của project.

## Cân chỉnh đen/trắng, phiên bản 0.3.1

- Chạy lại ảnh manga người dùng cung cấp, **900×1350 → 3600×5400**, bằng weights MangaScale GFISRV2 thật, GTX 1050 Ti 4 GiB, FP32, tile 1024. Cấu hình mới qua `Backend.process` tạo output đọc được, **0 OOM/retry**; xác minh SHA-256 ảnh gốc không đổi.
- Một vùng giày trong ảnh gốc có mọi pixel RGB=(25,25,25). Khi tắt cân chỉnh, model tạo chấm sáng ở vùng này, có cả khi chạy trực tiếp mạng gốc với RGB và FP16. Cấu hình FP32 tile 1024 trước sửa có mức xám output trung bình **18.75/255**, tối đa **253/255**. Bật `enhance_contrast` gốc nhận black=25, white=255: vùng đó còn trung bình **0.03/255**, tối đa **2/255**. Đây là bằng chứng trên một ảnh, không khẳng định mọi trang đều được cải thiện hoặc khớp từng pixel với ảnh mẫu.
- Eula và Ora chạy weights thật với ảnh thử grayscale **64×80**, tạo output lần lượt **256×320** và **128×160**. Theo dõi lời gọi xác nhận hàm cân chỉnh thực sự chạy trước AI, output preprocessing hữu hạn trong [0,1].
- Chạy thêm ảnh màu **64×80** qua MangaScale và Ora: xác nhận hàm cân chỉnh không được gọi, output đúng kích thước và đọc được. Cả năm lượt inference đều dùng FP32 và không OOM/retry; chưa thử bản cập nhật trên T4.
- Test hồi quy cho cả ba model: kết quả của workflow cũ không được resume nhầm sau thay đổi, không bị ghi đè khi chưa bật overwrite, chạy lại được với overwrite và tiếp tục resume được với cấu hình mới. Test xác minh cờ cân chỉnh đi vào settings JSON; workflow MangaJaNai gốc giữ nguyên.
- Vẫn giữ cách upstream báo lỗi NaN trên một số ảnh một màu. Không sửa thuật toán cân chỉnh hoặc tự thay model để che lỗi.

## Chưa xác minh

- Chưa mở phiên Colab GPU thực, chưa mount Google Drive thật, chưa chạy T4/L4/A100 hoặc FP16/BF16 trên các GPU đó.
- Chưa chạy batch hàng nghìn trang manga, ngắt mount thật hoặc quota Drive đầy thật. OOM GPU thật đã được thử bằng giới hạn allocator như mô tả ở trên.
- Với 16 model MangaJaNai/IllustrationJaNai, đã kiểm chứng archive/hash cho cả 16 và inference bốn model nêu trên. Eula/Ora/GFISRV2 đã chạy inference thật; chưa chạy inference 12 model còn lại của MangaJaNai/IllustrationJaNai.
- Không cam kết cùng pixel giữa các tile size/precision khác nhau.
- Default Colab Python 3.13 không tương thích `chainner_ext 0.3.10`; chọn runtime **2026.07 / Python 3.12**. Đây là giới hạn đã kiểm tra qua PyPI wheels và source bindings, không phải lỗi được che bằng một upscaler khác.

## Tái lập

```bash
python -m pytest -q
python scripts/smoke_test.py --scale 2
python scripts/smoke_test.py --scale 3
python -m manga_colab run --input .runtime/smoke/input \
  --output .runtime/smoke/downscale --models .runtime/smoke/models \
  --scale 1.5 --stage .runtime/stage
```

Weights, ảnh, backend logs và report có đường dẫn máy local đều nằm trong thư mục ignored `.runtime/`, không xuất bản lên GitHub.
