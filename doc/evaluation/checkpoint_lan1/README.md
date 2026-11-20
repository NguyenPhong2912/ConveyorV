# Phép đo bổ sung trên checkpoint lần 1

Các tệp trong thư mục này được đo ngày 20/11/2026 trên **checkpoint lần 1** (`conveyor_yolo26_30ep_best.pt`, 30 epoch, chia ngẫu nhiên theo ảnh), không phải checkpoint cuối cùng `conveyor_qc_yolo26n.pt`, vì tệp trọng số của lần 2 không nằm trên Git. Báo cáo dùng chúng ở các mục 7.4, 7.5 và 6.4.

| Tệp | Nội dung | Lệnh tạo |
| :--- | :--- | :--- |
| `threshold_sweep.csv` | Precision, Recall, F1 theo ngưỡng conf (val, test) | `python scripts/report_experiments.py sweep` |
| `confusion_counts.json` | Ma trận nhầm lẫn (đếm hộp) ở conf = 0,001 và 0,45 | `... confusion` |
| `demo_tracks.json` | Phán quyết theo track trên video demo, ảnh vào 640 và 480 px | `... tracking` |
| `speed_cpu.json` | Thời gian predict/track mỗi khung trên laptop i7-8565U | `... speed` |
| `pipeline_benchmark_20261120.json` | Tốc độ cả chuỗi của worker (150 khung) | chương trình đo cả chuỗi của lần 1 |
| `scenecut_check.json` | Tác dụng của `SceneCutDetector` lên phán quyết trên video demo | `python scripts/scenecut_check.py` |

Muốn đo lại cho checkpoint lần 2: `python scripts/report_experiments.py all --model models/conveyor_qc_yolo26n.pt --video raw_videos/demo_conveyor_testset.mp4` (kết quả vào `doc/evaluation/extra/`).
