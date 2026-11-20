# ConveyorVision — YOLO26 pilot

Hệ thống phân loại sản phẩm đạt/lỗi trên băng chuyền, dùng Ultralytics YOLO26.

## Cấu trúc

```
conveyorvision/
  data/
    data.yaml                  # file cấu hình duy nhất: khai báo đường dẫn tuyệt đối + tên lớp
    images/, labels/           # 2.650 ảnh thật đã chia train (1.866) / val (385) / test (399) — sinh bởi prepare_dataset.py
    split_manifest.csv         # ảnh nào thuộc tập nào, thuộc nhóm gần trùng nào
    split_report.json          # thống kê chia dữ liệu & rò rỉ
    manifest.csv               # sinh ra bởi video_io.py — KHÔNG commit video gốc
  doc/                         # tài liệu thiết kế, mindmap, báo cáo tiến độ
  notebooks/
    ConveyorVision_YOLO_Training.ipynb  # notebook huấn luyện & trực quan hóa biểu đồ
  src/conveyorvision/
    video_io.py                # trích frame từ video bằng OpenCV
    dataset_manager.py         # quản lý, remap 2 class và chuẩn hóa dữ liệu
    train_pilot.py             # script huấn luyện pilot qua CLI
    evaluate.py                # đánh giá độc lập mô hình trên tập test/val
    inference_conveyor.py      # pipeline kiểm định & đếm sản phẩm thời gian thực
  scripts/
    prepare_dataset.py         # chuẩn hoá 3 dataset Roboflow + chia theo nhóm chống rò rỉ
    make_demo_videos.py        # video demo từ ảnh thật của tập test
  models/                      # conveyor_qc_yolo26n.pt (QC Đạt/Lỗi), yolo26n.pt (COCO)
  tests/                       # bộ unit test tự động (pytest)
  reports/                     # log kết quả kiểm định CSV và báo cáo
```

## Bước 0 — cài môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Bước 1 — trích frame từ video

Bỏ video pilot (2-3 video ngắn) vào một thư mục, ví dụ `raw_videos/`, rồi chạy:

```bash
python src/conveyorvision/video_io.py \
  --video raw_videos/video1.mp4 \
  --out_dir data/images/train \
  --fps 2
```

Script sẽ lưu ảnh vào `data/images/train/` và ghi/append vào `data/manifest.csv`
(video_id, frame_idx, timestamp_s, image_path) — dùng file này để sau tách
train/val/test theo **video_id**, tránh rò rỉ dữ liệu giữa các frame liền kề.

Chạy lại với video khác, đổi `--out_dir data/images/val` hoặc `--out_dir data/images/test`
cho các phần video độc lập (không trộn frame của cùng một video vào các tập khác nhau).

## Bước 2 — Tải ảnh thật và chia Train / Val / Test chống rò rỉ

Dữ liệu là **2.666 ảnh thật** từ 3 dataset công khai trên Roboflow, chuẩn hoá về 2 lớp theo **tên** lớp:

| Dataset Roboflow | Ảnh | Nội dung | Ánh xạ |
| :--- | ---: | :--- | :--- |
| `anky06/conveyor-belt-eafpe-bgzzm` | 1.740 | Frame từ một video băng chuyền phân loại rác thải | `bottle` → `pass` |
| `bayo-et081/bottle-detection-qti1t` | 706 | Ảnh chai nhựa nguyên vẹn / bị bóp méo | `ok` → `pass`, `defect` → `defect` |
| `anky06/bottle_defects-tpmcq-kijtp` | 220 | Ảnh chai tốt / bị móp | `good` → `pass`, `defective` → `defect` |

Lớp `defect` trong dữ liệu này nghĩa là **chai bị bóp méo, móp, biến dạng** (không có nhãn lỗi nắp hay nhãn mác).

```bash
# 1) Tải 3 dataset vào raw_datasets/ (cần API key Roboflow, tài khoản miễn phí)
export ROBOFLOW_API_KEY=...
python src/conveyorvision/dataset_manager.py --download-all

# 2) Chuẩn hoá nhãn và chia train/val/test theo nhóm → data/
python scripts/prepare_dataset.py
```

Split ngẫu nhiên gốc của Roboflow để **63 ảnh val/test có bản gần trùng trong train**. Vì vậy `prepare_dataset.py` chia lại như sau:
- Video băng chuyền chia **theo thời gian**: 70% đầu → train, 15% → val, 15% cuối → test, bỏ 16 frame ở các điểm cắt.
- Ảnh tĩnh được gộp nhóm theo **pHash** (Hamming ≤ 6, kể cả khác dataset) rồi chia **nguyên nhóm**, phân tầng theo nguồn và theo có/không có defect.

Kết quả: **1.866 / 385 / 399** ảnh (70/15/15). Còn đúng 1 cặp gần trùng giữa test và train: hai frame cách nhau hơn 1.000 frame, băng chuyền trông giống nhau, không phải rò rỉ chuỗi.

## Bước 3 — Khai báo data/data.yaml

Dự án quản lý cấu hình tập dữ liệu tại `data/data.yaml` với `path: data` tương đối chuẩn Git. Khi chạy qua `evaluate.py`, `train_pilot.py` hoặc Notebook, hệ thống tự động sinh cấu hình runtime cô lập (`runs/runtime_data.yaml`) chứa đường dẫn tuyệt đối cho Ultralytics xử lý mà không làm thay đổi file `data.yaml` trong kho Git.

## Bước 4 — Huấn luyện mô hình (Training)

```bash
python src/conveyorvision/train_pilot.py --model models/yolo26n.pt --epochs 80 --patience 20 \
    --cache ram --batch 16 --device 0 --project runs/train --name conveyor_yolo26_groupsplit
cp runs/detect/runs/train/conveyor_yolo26_groupsplit/weights/best.pt models/conveyor_qc_yolo26n.pt
```

Notebook `notebooks/ConveyorVision_YOLO_Training.ipynb` (Colab) vẫn dùng được để thử nghiệm.

## Bước 5 — Đánh giá trên tập Test và xuất số liệu cho dashboard

```bash
python src/conveyorvision/evaluate.py --model models/conveyor_qc_yolo26n.pt --split test --device 0 \
    --export doc/evaluation --train-run runs/detect/runs/train/conveyor_yolo26_groupsplit
```

Lệnh này ghi `doc/evaluation/metrics.json` và các biểu đồ. Màn hình **Mô hình & Dữ liệu** chỉ hiển thị số liệu từ file này, không có số liệu viết cứng.

## Bước 6 — Khởi chạy Giao diện Dashboard Trực Quan (Web GUI) & CLI

### Cách 1: Giao diện Web Dashboard Trực quan (Khuyên dùng)
Hệ thống cung cấp giao diện Web Dashboard (FastAPI + WebSocket, không phụ thuộc CDN) gồm 6 màn hình, theo luồng **Tổng quan → Vận hành → Lịch sử**:
* **Tổng quan:** danh sách *Sẵn sàng vận hành* (thư viện YOLO, GPU, trọng số COCO, checkpoint QC, video nguồn) kèm hướng dẫn khắc phục, cùng thống kê và các phiên gần nhất.
* **Vận hành:** thiết lập phiên theo 3 bước (nguồn video/webcam + tải lên kéo thả → chế độ kiểm định → tham số nâng cao). Nút *Bắt đầu* chỉ bật khi đủ điều kiện và ghi rõ lý do nếu chưa. Khi chạy: video MJPEG lớn, chỉ số trực tiếp, nhật ký sản phẩm cập nhật tại chỗ, còi cảnh báo; khi kết thúc hiện ngay bảng tổng kết phiên.
* **Lịch sử phiên:** tìm kiếm, xem chi tiết từng sản phẩm, tải CSV, xoá phiên.
* **Mô hình & Dữ liệu:** chỉ số đánh giá (kèm cảnh báo rò rỉ dữ liệu cận trùng), biểu đồ, danh sách checkpoint (COCO / QC) và phân bổ tập dữ liệu.
* **Cài đặt:** giá trị mặc định cho phiên mới (tracker, ngưỡng tin cậy, lost buffer, số frame quan sát, CSV, còi).
* **Tài liệu:** quy trình vận hành, phím tắt và các tài liệu PDF.

Chế độ **Chai & Dị vật** cần trọng số COCO (ví dụ `models/yolo26n.pt`); chế độ **QC Đạt/Lỗi** cần checkpoint tự huấn luyện trong `models/`.
Phím tắt ở màn hình Vận hành: `Space` tạm dừng/tiếp tục, `S` chụp hiện trường, `F` toàn màn hình.

Khởi chạy server giao diện:
```bash
python run_app.py
# Hoặc tùy biến cổng: python run_app.py --port 8080 --host 0.0.0.0
```
Truy cập ngay trên trình duyệt tại: **`http://localhost:8000`**

### Cách 2: Chạy kiểm định qua dòng lệnh CLI
Pipeline theo vết (Object Tracking với ByteTrack), cơ chế **Multi-frame Voting** chống phán đoán vội ở mép khung hình, đếm chính xác sản phẩm không bị trùng lặp, cảnh báo phế phẩm và xuất log CSV:
```bash
# Chạy trên file video băng chuyền với tracker ByteTrack
python src/conveyorvision/inference_conveyor.py --source raw_videos/demo_bottles_testset.mp4 --model models/conveyor_qc_yolo26n.pt --tracker bytetrack.yaml

# Chạy trực tiếp từ camera / webcam
python src/conveyorvision/inference_conveyor.py --source 0 --model models/conveyor_qc_yolo26n.pt
```

## 📊 Kết Quả Đánh Giá Thực Nghiệm (tập Test, chia theo nhóm)

YOLO26n huấn luyện từ trọng số COCO, dừng sớm ở epoch 67 (tốt nhất: epoch 47), mất 55 phút trên Quadro RTX 6000 dùng chung.
Đánh giá trên **399 ảnh test, 802 box**, mô hình chưa từng thấy khi huấn luyện:

| Lớp | Ảnh | Box | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: |
| **Tất cả** | 399 | 802 | 70.3% | 77.8% | **78.8%** | 54.5% |
| `pass` | 294 | 724 | 68.1% | 72.2% | 75.6% | 51.3% |
| `defect` | 66 | 78 | 72.5% | **83.3%** | **82.1%** | 57.6% |

* **Tốc độ đo thật (640px):** GPU Quadro RTX 6000 **~63 FPS** (15.9 ms/ảnh); CPU Intel Xeon Gold 5218R **~19 FPS** (52.3 ms/ảnh).
* **Giới hạn cần biết:** tại ngưỡng conf=0.25 trong ma trận nhầm lẫn, **41% chai lỗi bị nhận nhầm thành `pass`** (59% đúng, 0% bỏ sót). Nguyên nhân là dữ liệu mất cân bằng: train chỉ có 366 box defect so với 3.611 box pass. Cơ chế Multi-frame Voting của ứng dụng giảm bớt rủi ro này: chỉ cần một frame thấy lỗi với conf ≥ 0.5 là chốt DEFECT. Muốn dùng thật cần thêm ảnh chai lỗi chụp tại hiện trường.
* **Khác biệt miền (domain gap):** dữ liệu là ảnh chai đơn lẻ và băng chuyền rác thải. Trên video dây chuyền chiết rót dày đặc (stock Mixkit) mô hình gần như không phát hiện được, nên cần fine-tune bằng video từ camera thật của trạm.
* Số liệu đầy đủ: [`doc/evaluation/metrics.json`](doc/evaluation/metrics.json) · giải thích: [`doc/evaluation/README.md`](doc/evaluation/README.md).

## 🎬 Video demo

`python scripts/make_demo_videos.py` tạo 2 video từ ảnh thật **của tập test** vào `raw_videos/`:
`demo_bottles_testset.mp4` (chai pass/defect) và `demo_conveyor_testset.mp4` (đoạn cuối video băng chuyền).
Video stock dây chuyền thật (giấy phép Mixkit, miễn phí): [Beer bottle production line](https://mixkit.co/free-stock-video/beer-bottle-production-line-in-the-factory-33003/).

## 📥 Trọng Số Mô Hình (.pt)

File `*.pt` không lưu trên Git (xem [`.gitignore`](.gitignore)). Khi clone mới:
1. Tự huấn luyện lại theo Bước 2 → 4, rồi chép `best.pt` thành `models/conveyor_qc_yolo26n.pt`.
2. Hoặc nhận file `conveyor_qc_yolo26n.pt` có sẵn và đặt vào `models/`.
3. Chế độ *Chai & Dị vật* dùng `models/yolo26n.pt` (COCO). Tải bằng: `cd models && python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"`



\\\

## 🔧 Phát triển & Đóng góp

### Cài đặt môi trường phát triển

\\\ash
# Clone repository
git clone https://github.com/anky06-ky/ConveyorVision.git
cd ConveyorVision

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt package ở chế độ development
pip install -e ".[dev]"
\\\

### Cấu hình Environment Variables

1. Copy \.env.example\ thành \.env\:
   \\\ash
   cp .env.example .env   # Linux/Mac
   copy .env.example .env # Windows
   \\\

2. Cập nhật \ROBOFLOW_API_KEY\ nếu cần download dataset tự động

### Chạy Tests

\\\ash
# Chạy tất cả tests
pytest tests/ -v

# Chạy với coverage report
pytest tests/ -v --cov=conveyorvision --cov-report=html

# Xem coverage report
# Mở htmlcov/index.html trong trình duyệt
\\\

### Code Quality

\\\ash
# Format code với Black
black src/ tests/

# Kiểm tra với Flake8
flake8 src/ tests/ --max-line-length=100

# Type checking với MyPy
mypy src/conveyorvision --ignore-missing-imports
\\\

### Đóng góp

Xem hướng dẫn chi tiết tại [CONTRIBUTING.md](CONTRIBUTING.md)

## 📦 Cài đặt như Python Package

Project có thể được cài đặt như một Python package chuẩn:

\\\ash
# Từ PyPI (sau khi publish)
pip install conveyorvision

# Hoặc từ source
git clone https://github.com/anky06-ky/ConveyorVision.git
cd ConveyorVision
pip install -e .
\\\

Sau khi cài đặt, có thể sử dụng các CLI commands:

\\\ash
# Huấn luyện mô hình
conveyor-train --epochs 30 --batch 16

# Đánh giá mô hình
conveyor-eval --split test

# Chạy kiểm định
conveyor-inspect --source raw_videos/demo.mp4

# Khởi chạy Web Dashboard
conveyor-web --port 8000
\\\

## 📄 License

Dự án được phân phối dưới MIT License. Xem [LICENSE](LICENSE) để biết thêm chi tiết.

## 🙏 Credits

- **YOLO26** by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Dataset** từ [Roboflow](https://roboflow.com)
- **ByteTrack** cho Object Tracking

---

Made with ❤️ by **ConveyorVision Team**

[![GitHub stars](https://img.shields.io/github/stars/anky06-ky/ConveyorVision.svg?style=social&label=Star)](https://github.com/anky06-ky/ConveyorVision)
