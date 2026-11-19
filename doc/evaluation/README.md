# Đánh giá mô hình ConveyorVision QC (YOLO26n)

Mọi số liệu dưới đây lấy từ [`metrics.json`](metrics.json). File này do lệnh sau sinh ra, và dashboard đọc trực tiếp từ đó:

```bash
python src/conveyorvision/evaluate.py --model models/conveyor_qc_yolo26n.pt --split test --device 0 \
    --export doc/evaluation --train-run runs/detect/runs/train/conveyor_yolo26_groupsplit
```

## 1. Dữ liệu

- 2.666 ảnh thật từ 3 dataset Roboflow. Sau khi bỏ 16 frame ở các điểm cắt video còn **2.650 ảnh**: train 1.866 / val 385 / test 399.
- Chia **theo nhóm** bằng `scripts/prepare_dataset.py`:
  - video băng chuyền chia theo thời gian;
  - ảnh tĩnh gộp nhóm gần trùng bằng pHash (Hamming ≤ 6) rồi chia nguyên nhóm.
- Rò rỉ (số ảnh val/test có bản gần trùng trong train): split ngẫu nhiên gốc của Roboflow **63** → split theo nhóm **1**. Ảnh còn lại là frame 1741 so với frame 661, băng chuyền trông giống nhau, không phải chuỗi liền kề.
- `defect` = chai bị bóp méo / móp / biến dạng. `pass` = chai nguyên vẹn, gồm cả chai trên băng chuyền phân loại rác thải.

## 2. Huấn luyện

YOLO26n khởi tạo từ trọng số COCO, imgsz 640, batch 16, `cls=1.2`, early stopping (patience 20).
Huấn luyện dừng ở epoch 67, tốt nhất ở epoch 47, mất 55 phút trên Quadro RTX 6000 dùng chung với người khác.

## 3. Kết quả trên tập Test (399 ảnh, 802 box)

| Lớp | Ảnh | Box | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | ---: | ---: | :---: | :---: | :---: | :---: |
| **Tất cả** | 399 | 802 | 70.3% | 77.8% | **78.8%** | 54.5% |
| `pass` | 294 | 724 | 68.1% | 72.2% | 75.6% | 51.3% |
| `defect` | 66 | 78 | 72.5% | 83.3% | 82.1% | 57.6% |

**Ma trận nhầm lẫn tại conf = 0.25** (chuẩn hoá theo nhãn thật):

| Nhãn thật | Đúng | Nhầm sang lớp kia | Bỏ sót (background) |
| :--- | :---: | :---: | :---: |
| `pass` | 94.6% | 3.3% | 2.1% |
| `defect` | 59.0% | **41.0%** | 0.0% |

Mô hình hầu như **không bỏ sót** chai, nhưng **41% chai lỗi bị gán nhầm là `pass`** ở ngưỡng này.
Nguyên nhân chính là mất cân bằng lớp: train có 366 box defect so với 3.611 box pass.
Trong ứng dụng, cơ chế Multi-frame Voting giảm rủi ro này: một frame thấy lỗi với conf ≥ 0.5, hoặc ≥ 2 frame thấy lỗi, là chốt DEFECT.

## 4. Tốc độ (640px)

| Thiết bị | ms/ảnh (tiền xử lý + suy luận + hậu xử lý) | FPS |
| :--- | :---: | :---: |
| GPU Quadro RTX 6000 | 15.9 | ~63 |
| CPU Intel Xeon Gold 5218R @ 2.10GHz | 52.3 | ~19 |

## 5. Giới hạn và hướng cải thiện

1. **Thêm ảnh chai lỗi:** lớp `defect` chỉ chiếm 9.4% số box.
2. **Domain gap:** dữ liệu là ảnh chai đơn lẻ và băng chuyền rác thải. Trên video dây chuyền chiết rót dày đặc, chai nhìn từ trên xuống, mô hình gần như không phát hiện được. Cần fine-tune bằng video từ camera thật của trạm.
3. **Loại lỗi:** dữ liệu chỉ có lỗi biến dạng, chưa có lỗi nắp, nhãn mác hay thể tích.

## 6. Biểu đồ

| File | Nội dung |
| :--- | :--- |
| `results.png`, `results.csv` | Loss và mAP theo từng epoch |
| `test_confusion_matrix_normalized.png` | Ma trận nhầm lẫn trên tập test |
| `test_BoxPR_curve.png` | Đường cong Precision–Recall trên tập test |
| `test_batch0_pred.jpg` | Dự đoán mẫu trên ảnh test |
