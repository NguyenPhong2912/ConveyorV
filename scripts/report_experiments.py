"""Các phép đo bổ sung dùng trong báo cáo đồ án, chạy lại được từ dòng lệnh.

Mỗi lệnh con ghi một tệp kết quả vào doc/evaluation/extra (đổi bằng --out; đổi trọng số bằng --model):

    python scripts/report_experiments.py sweep       # P/R/F1 theo ngưỡng conf trên val và test
    python scripts/report_experiments.py confusion   # ma trận nhầm lẫn (đếm hộp) ở conf=0.001 và 0.45
    python scripts/report_experiments.py tracking    # phán quyết từng track trên video demo, 640 px và 480 px
    python scripts/report_experiments.py speed       # thời gian predict/track mỗi khung trên CPU
    python scripts/report_experiments.py all

Chạy từ thư mục gốc của dự án. Ngưỡng và luật bỏ phiếu giống inference_conveyor.py.
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "models/conveyor_qc_yolo26n.pt"
DATA = "data/data.yaml"
VIDEO = "raw_videos/conveyor_demo.mp4"  # video demo dùng trong báo cáo (640 x 640, 10 khung/giây, 150 khung)
CLASS_NAMES = {0: "pass", 1: "defect"}


def _val(split, conf, plots, out):
    model = YOLO(MODEL)
    return model.val(data=DATA, split=split, imgsz=640, batch=16, conf=conf, plots=plots, verbose=False,
                     project=str(out / "_tmp_val"), name=f"{split}_{conf}", exist_ok=True, workers=0)


def sweep(out):
    """Precision, Recall, F1 của từng lớp tại các ngưỡng conf từ 0.10 đến 0.90 (IoU 0.5)."""
    rows = []
    for split in ("val", "test"):
        result = _val(split, 0.001, False, out).box
        for conf in np.round(np.arange(0.10, 0.901, 0.05), 2):
            i = int(round(conf * (len(result.px) - 1)))
            for k, cls in enumerate(result.ap_class_index):
                rows.append([split, conf, CLASS_NAMES[int(cls)], round(float(result.p_curve[k, i]), 4),
                             round(float(result.r_curve[k, i]), 4), round(float(result.f1_curve[k, i]), 4)])
    with open(out / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "conf", "class", "precision", "recall", "f1"])
        w.writerows(rows)
    print("đã ghi", out / "threshold_sweep.csv")


def confusion(out):
    """Ma trận nhầm lẫn [dự đoán, thật] của Ultralytics, hàng/cột cuối là nền (background)."""
    res = {}
    for split in ("val", "test"):
        for conf in (0.001, 0.45):
            r = _val(split, conf, True, out)
            res[f"{split}_conf{conf}"] = {"labels": ["pass", "defect", "background"],
                                          "matrix_pred_by_true": r.confusion_matrix.matrix.astype(int).tolist()}
    json.dump(res, open(out / "confusion_counts.json", "w", encoding="utf-8"), indent=1)
    print("đã ghi", out / "confusion_counts.json")


def tracking(out):
    """Chạy ByteTrack + luật bỏ phiếu (B=30, V=3, ngưỡng khoá DEFECT max(0.5, conf)) trên video demo."""
    conf_t, buffer, min_votes = 0.45, 30, 3
    res = {}
    for imgsz in (640, 480):
        model = YOLO(MODEL)
        cap = cv2.VideoCapture(VIDEO)
        tracks, k = {}, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            k += 1
            r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_t, imgsz=imgsz, verbose=False)[0]
            if r.boxes is not None and r.boxes.id is not None:
                for tid, c, s in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    t = tracks.setdefault(tid, dict(first=k, last=k, detections=0, pass_votes=0, defect_votes=0,
                                                    max_defect_conf=0.0, state="INSPECTING", decided_at=None))
                    t["last"] = k
                    t["detections"] += 1
                    if c == 1:
                        t["defect_votes"] += 1
                        t["max_defect_conf"] = max(t["max_defect_conf"], s)
                    else:
                        t["pass_votes"] += 1
                    if t["state"] == "INSPECTING" and (t["defect_votes"] >= 2 or
                                                       (t["defect_votes"] >= 1 and t["max_defect_conf"] >= max(0.5, conf_t))):
                        t["state"], t["decided_at"] = "DEFECT", k
            for t in tracks.values():
                if t["state"] == "INSPECTING" and k - t["last"] >= buffer:
                    t["state"] = "PASS" if t["pass_votes"] + t["defect_votes"] >= min_votes else "INVALID"
                    t["decided_at"] = k
        cap.release()
        for t in tracks.values():
            if t["state"] == "INSPECTING":  # hết video: chốt ngay
                t["state"] = "PASS" if t["pass_votes"] + t["defect_votes"] >= min_votes else "INVALID"
                t["decided_at"] = f"{k} (hết video)"
        res[str(imgsz)] = {"frames": k, "tracks": {str(tid): t for tid, t in tracks.items()}}
    # số phát hiện điểm thấp (0,1 đến 0,45) bị ngưỡng conf = 0,45 loại trước khi tới ByteTrack
    model, cap, total, low = YOLO(MODEL), cv2.VideoCapture(VIDEO), 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scores = model.predict(frame, conf=0.1, imgsz=640, verbose=False)[0].boxes.conf.tolist()
        total += len(scores)
        low += sum(1 for s in scores if s < conf_t)
    cap.release()
    res["low_score_640"] = {"detections_conf_ge_0.1": total, "detections_below_0.45": low}
    json.dump(res, open(out / "demo_tracks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("đã ghi", out / "demo_tracks.json")


def speed(out, warmup=5, frames=60):
    """Thời gian trung bình mỗi khung của predict() và track() (không gồm giải mã video)."""
    cap = cv2.VideoCapture(VIDEO)
    imgs = []
    while len(imgs) < warmup + frames:
        ok, frame = cap.read()
        if not ok:
            break
        imgs.append(frame)
    cap.release()
    res = {"cpu": platform.processor(), "frames_timed": len(imgs) - warmup, "unit": "ms/khung", "results": {}}
    try:
        from ultralytics.utils.torch_utils import get_cpu_info
        res["cpu"] = get_cpu_info()
    except Exception:
        pass
    for imgsz in (640, 480):
        for mode in ("predict", "track"):
            model = YOLO(MODEL)
            times = []
            for n, frame in enumerate(imgs):
                t0 = time.perf_counter()
                if mode == "predict":
                    model.predict(frame, conf=0.45, imgsz=imgsz, verbose=False)
                else:
                    model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.45, imgsz=imgsz, verbose=False)
                if n >= warmup:
                    times.append((time.perf_counter() - t0) * 1000)
            res["results"][f"{mode}_{imgsz}"] = {"ms": round(float(np.mean(times)), 1), "fps": round(1000 / float(np.mean(times)), 1)}
    json.dump(res, open(out / "speed_cpu.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("đã ghi", out / "speed_cpu.json")


def main():
    global MODEL, VIDEO, DATA
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sweep", "confusion", "tracking", "speed", "all"])
    parser.add_argument("--out", default="doc/evaluation/extra", help="thư mục ghi kết quả")
    parser.add_argument("--model", default=MODEL, help="tệp trọng số cần đo")
    parser.add_argument("--video", default=VIDEO, help="video dùng cho phép đo theo vết và tốc độ")
    parser.add_argument("--data", default=DATA, help="tệp data.yaml")
    args = parser.parse_args()
    MODEL, VIDEO, DATA = args.model, args.video, args.data
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (["sweep", "confusion", "tracking", "speed"] if args.command == "all" else [args.command]):
        globals()[name](out)
    tmp = out / "_tmp_val"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
