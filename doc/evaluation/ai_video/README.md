# Đếm chai trên video băng chuyền do AI tạo (mục 7.7 của báo cáo)

Video: `video_bang_truyen.mp4` do nhóm tự tạo bằng công cụ AI sinh video, là cảnh tổng hợp chứ không phải quay tại một băng chuyền thật, nên kết quả chỉ cho biết mô hình phản ứng thế nào với một cảnh chưa từng thấy, không đo hiệu năng trên dây chuyền thật. Thông số: 30 giây, 1280 × 720, 24 khung hình mỗi giây, 721 khung. `videophanloai_nhe.mp4` là cùng cảnh được nén lại (30 khung hình mỗi giây, 901 khung). Hai tệp không nằm trên Git; đặt vào `raw_videos/`.

| Tệp | Nội dung |
|---|---|
| `ground_truth.json` | Đáp án đếm thủ công: 8 chai và 10 vật không phải chai (hai can sữa tính là hai vật), kèm thời điểm vào và ra khỏi khung hình (giây). |
| `qc_480.json`, `qc_640.json` | Checkpoint lần 1, chế độ QC Đạt / Lỗi, ảnh vào 480 px và 640 px, trên `video_bang_truyen.mp4`. |
| `coco_480.json`, `coco_640.json` | Mô hình COCO `yolo26n.pt`, chế độ Chai & Dị vật khác (chế độ mặc định của web), 480 px và 640 px. |
| `nhe_qc_480.json`, `nhe_coco_480.json` | Như trên, trên `videophanloai_nhe.mp4`. |
| `labels_*.json` | Nhãn tay của từng track: mã vật trong đáp án mà hộp ôm (B1–B8, O1–O9), `WIDE` nếu hộp phủ gần cả khung, `NONE` nếu không ôm vật nào. |
| `snapshots/` | Khung hình do InspectionWorker vẽ, dùng cho Hình 7.10. |

Checkpoint lần 1: SHA-256 `b9c06fd30e896c168f3d656dbed521020984046e06cb7f44060a8435e9f1e59f` (trùng với `checkpoint_lan1/pipeline_benchmark_20261120.json`).

Chạy lại (từ thư mục gốc), hai bước:

```
python scripts/video_count_eval.py --video raw_videos/video_bang_truyen.mp4 \
    --model models/conveyor_qc_yolo26n.pt --mode bottle_inspection --imgsz 480 \
    --gt doc/evaluation/ai_video/ground_truth.json --out doc/evaluation/ai_video/qc_480.json

python scripts/video_count_eval.py --eval-only doc/evaluation/ai_video/qc_480.json \
    --gt doc/evaluation/ai_video/ground_truth.json --labels doc/evaluation/ai_video/labels_qc_480.json
```

Lưu ý: đáp án và nhãn tay do người đếm bằng mắt (đáp án đã được đối chiếu với một lần đếm độc lập thứ hai), nhóm nên xem lại trước khi nộp. Ghép track với vật theo thời gian sai khi hai vật cùng có mặt, nên kết quả trong báo cáo luôn dùng nhãn tay. Chạy lại worker trên máy khác có thể cho số track khác một chút; khi đó phải gán lại nhãn tay.
