"""Đo sai số đếm chai trên một video băng chuyền, dùng đúng InspectionWorker của ứng dụng web.

Video đã dùng trong báo cáo (video_bang_truyen.mp4) do nhóm tự tạo bằng công cụ AI sinh video, không phải cảnh quay thật.

Cách chạy (từ thư mục gốc của dự án), hai bước:

    # Bước 1: chạy worker trên video, ghi các track và khung hình tốt nhất của mỗi track
    python scripts/video_count_eval.py --video raw_videos/video_bang_truyen.mp4 \
        --model models/conveyor_qc_yolo26n.pt --mode bottle_inspection --imgsz 480 \
        --gt doc/evaluation/ai_video/ground_truth.json --out doc/evaluation/ai_video/qc_480.json

    # Bước 2: kiểm tra bằng mắt hộp của từng track rồi ghi nhãn vào tệp labels (track -> mã vật trong đáp án,
    # hoặc WIDE nếu hộp phủ gần cả khung, hoặc NONE nếu hộp không ôm vật nào), sau đó tính lại:
    python scripts/video_count_eval.py --eval-only doc/evaluation/ai_video/qc_480.json \
        --gt doc/evaluation/ai_video/ground_truth.json --labels doc/evaluation/ai_video/labels_qc_480.json

Với --mode bottle_and_other (chế độ mặc định của web) hãy truyền trọng số COCO yolo26n.pt qua --model.
Worker chạy ở chế độ không lặp video với tham số mặc định của web (conf 0,45, ByteTrack, B = 30, V = 3).
Phiên kiểm định được ghi vào một thư mục tạm, không động tới reports/ của dự án.
Không có nhãn tay thì chỉ ghép được theo thời gian, cách này sai khi hai vật cùng có mặt trong một lúc;
vì vậy kết quả đưa vào báo cáo luôn dùng nhãn tay.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIGHT_MAX_AREA = 0.35   # hộp chiếm tối đa 35% khung hình: coi là hộp ôm một vật
WIDE_MIN_AREA = 0.50    # hộp chiếm từ 50% khung hình trở lên: hộp phủ gần cả khung


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def box_area(xyxyn):
    return max(0.0, xyxyn[2] - xyxyn[0]) * max(0.0, xyxyn[3] - xyxyn[1])


def evaluate(result, gt, labels):
    """So số track được đếm là chai (PASS hoặc DEFECT) với đáp án thủ công, theo nhãn tay của từng track."""
    objs = {o["id"]: o for o in gt["objects"]}
    bottles = [o["id"] for o in gt["objects"] if o["kind"] == "bottle"]
    tracks = result["tracks"]
    counted = [t for t in tracks if t["state"] in ("PASS", "DEFECT")]
    per_obj = {b: [] for b in bottles}
    fp_other, fp_wide, fp_none = {}, [], []
    for t in counted:
        lab = labels.get(str(t["track_id"]), "NONE")
        if lab in per_obj:
            per_obj[lab].append(t["track_id"])
        elif lab in objs:
            fp_other.setdefault(lab, []).append(t["track_id"])
        elif lab == "WIDE":
            fp_wide.append(t["track_id"])
        else:
            fp_none.append(t["track_id"])
    found = [b for b in bottles if per_obj[b]]
    dup_extra = sum(len(v) - 1 for v in per_obj.values() if len(v) > 1)
    n_fp = sum(len(v) for v in fp_other.values()) + len(fp_wide) + len(fp_none)
    out = {
        "gt_bottles": len(bottles),
        "counted_tracks": len(counted),
        "counted_pass": sum(1 for t in counted if t["state"] == "PASS"),
        "counted_defect": sum(1 for t in counted if t["state"] == "DEFECT"),
        "count_error": len(counted) - len(bottles),
        "bottles_found": found,
        "bottles_missed": [b for b in bottles if b not in found],
        "extra_tracks_on_same_bottle": dup_extra,
        "false_tracks": n_fp,
        "false_tracks_on_other_objects": fp_other,
        "false_tracks_wide_box": fp_wide,
        "false_tracks_on_background": fp_none,
        "recall_bottle": round(len(found) / len(bottles), 3),
        "precision_tracks": round((len(counted) - n_fp - dup_extra) / len(counted), 3) if counted else None,
        "assignment": per_obj,
    }
    other_tracks = [t for t in tracks if t["state"] == "OTHER"]
    if other_tracks:
        kind = lambda t: objs.get(labels.get(str(t["track_id"]), ""), {}).get("kind")
        out["other_tracks"] = len(other_tracks)
        out["other_tracks_on_real_other_objects"] = [t["track_id"] for t in other_tracks if kind(t) == "other"]
        # chai thật có hộp ôm riêng nhưng bị gọi sai tên (ví dụ "bird", "cell phone") nên không được đếm là chai
        seen = sorted({labels[str(t["track_id"])] for t in other_tracks if kind(t) == "bottle"})
        out["bottles_seen_but_named_other"] = seen
        out["bottles_seen_by_any_class"] = sorted(set(found) | set(seen))
    return out


def run_worker(args):
    video, model = str(Path(args.video).resolve()), str(Path(args.model).resolve())
    snap_dir = Path(args.snap_dir).resolve() if args.snap_dir else None
    if snap_dir:
        snap_dir.mkdir(parents=True, exist_ok=True)
    work = tempfile.mkdtemp(prefix="conveyorvision_run_")  # worker ghi reports/ vào đây
    os.chdir(work)
    sys.path.insert(0, str(ROOT / "src"))
    import cv2
    from ultralytics import YOLO
    from conveyorvision.web.worker import worker

    # Bọc YOLO.track để ghi lại hộp của từng khung. Chỉ quan sát: kết quả trả về worker không đổi.
    frame_log = []
    orig_track = YOLO.track

    def logged_track(self, *a, **k):
        res = orig_track(self, *a, **k)
        dets = []
        try:
            b = res[0].boxes
            if b is not None and len(b):
                ids = b.id.int().tolist() if b.id is not None else [-1] * len(b)
                for i, (xy, c, sc) in enumerate(zip(b.xyxyn.tolist(), b.cls.int().tolist(), b.conf.tolist())):
                    dets.append((ids[i], c, round(sc, 3), [round(v, 4) for v in xy]))
        except Exception:
            pass
        frame_log.append(dets)
        return res

    YOLO.track = logged_track

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    snap_every = max(1, int(round(fps * args.snap_every)))

    t0 = time.time()
    res = worker.start(source=video, model_path=model, inspection_mode=args.mode, conf_thresh=args.conf,
                       tracker_config="bytetrack.yaml", imgsz=args.imgsz, loop_video=False,
                       lost_track_buffer=30, min_observation_votes=3, auto_save_csv=False)
    if not res.get("success"):
        sys.exit("không khởi chạy được: %s" % res)
    last_seq = -1
    while worker.state in ("RUNNING", "PAUSED"):
        with worker._frame_cond:
            worker._frame_cond.wait(timeout=0.3)
            seq, jpg, fidx = worker._frame_seq, worker.latest_jpeg, worker.frame_idx
        if snap_dir and seq != last_seq and jpg and fidx % snap_every == 0 and fidx > 0:
            (snap_dir / ("f%04d.jpg" % fidx)).write_bytes(jpg)
        last_seq = seq
    wall = time.time() - t0
    if worker.state == "ERROR":
        sys.exit("worker báo lỗi: %s" % worker.error_message)

    stat = {}
    for k, dets in enumerate(frame_log, start=1):
        for tid, _c, sc, xy in dets:
            st = stat.setdefault(tid, {"n": 0, "area_sum": 0.0, "area_max": 0.0, "best": (-1, 0, None)})
            a = box_area(xy)
            st["n"] += 1
            st["area_sum"] += a
            st["area_max"] = max(st["area_max"], a)
            if sc > st["best"][0]:
                st["best"] = (sc, k, xy)
    tracks = []
    for tid, th in sorted(worker.track_history.items()):
        st = stat.get(tid, {"n": 0, "area_sum": 0.0, "area_max": 0.0, "best": (0, 0, None)})
        tracks.append({"track_id": tid, "state": th["state"], "category": th["category"],
                       "display_name": th["display_name"], "first_ts": round(th["first_ts"], 2),
                       "last_ts": round(th["last_seen_ts"], 2), "boxes": st["n"],
                       "mean_area": round(st["area_sum"] / max(1, st["n"]), 3), "max_area": round(st["area_max"], 3),
                       "pass_votes": th["pass_votes"], "defect_votes": th["defect_votes"],
                       "other_votes": th["other_votes"], "max_conf": round(th["max_conf"], 2),
                       "best_frame": st["best"][1], "best_box": st["best"][2]})
    return {
        "video": Path(video).name, "video_sha256": sha256(video), "frames": total, "fps": round(fps, 3),
        "model": Path(model).name, "model_sha256": sha256(model), "mode": args.mode, "imgsz": args.imgsz,
        "conf": args.conf, "frames_processed": worker.frame_idx, "wall_seconds": round(wall, 1),
        "worker_avg_fps": round(worker.fps, 1), "tracks_total": len(tracks),
        "summary": {s: sum(1 for t in tracks if t["state"] == s) for s in ("PASS", "DEFECT", "OTHER", "INVALID")},
        "frames_with_wide_box": sum(1 for d in frame_log if any(box_area(x[3]) >= WIDE_MIN_AREA for x in d)),
        "frames_with_tight_box": sum(1 for d in frame_log if any(box_area(x[3]) < TIGHT_MAX_AREA for x in d)),
        "tight_box_frames": [k for k, d in enumerate(frame_log, start=1) if any(box_area(x[3]) < TIGHT_MAX_AREA for x in d)],
        "tracks": tracks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--model")
    ap.add_argument("--mode", default="bottle_inspection", choices=["bottle_inspection", "bottle_and_other"])
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--gt", default=None, help="tệp đáp án thủ công (ground_truth.json)")
    ap.add_argument("--labels", default=None, help="nhãn tay của từng track (JSON: {\"7\": \"B3\", \"1\": \"WIDE\"})")
    ap.add_argument("--out", help="tệp JSON kết quả")
    ap.add_argument("--snap-dir", default=None, help="lưu khung hình đã vẽ của worker vào thư mục này")
    ap.add_argument("--snap-every", type=float, default=0.5, help="giây giữa hai ảnh chụp")
    ap.add_argument("--eval-only", default=None, help="tính lại độ chính xác từ tệp kết quả đã có và --labels")
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8")) if args.gt else None
    labels = json.load(open(args.labels, encoding="utf-8")) if args.labels else None
    if args.eval_only:
        out = Path(args.eval_only)
        result = json.load(open(out, encoding="utf-8"))
    else:
        if not (args.video and args.model and args.out):
            ap.error("cần --video, --model và --out (hoặc dùng --eval-only)")
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_worker(args)
        print(result["mode"], result["imgsz"], result["video"], "| khung:", result["frames_processed"],
              "| track:", result["tracks_total"], result["summary"],
              "| khung có hộp phủ cả khung:", result["frames_with_wide_box"],
              "| khung có hộp ôm vật:", result["frames_with_tight_box"])
    if gt and labels is not None:
        result["manual_labels"] = labels
        result["evaluation"] = evaluate(result, gt, labels)
        e = result["evaluation"]
        print("đáp án %d chai | đếm %d (PASS %d, DEFECT %d) | tìm thấy %s | sót %s | thừa cùng chai %d | track sai %d "
              "(vật khác %s, hộp phủ cả khung %s, nền %s)" % (
                  e["gt_bottles"], e["counted_tracks"], e["counted_pass"], e["counted_defect"], e["bottles_found"],
                  e["bottles_missed"], e["extra_tracks_on_same_bottle"], e["false_tracks"],
                  e["false_tracks_on_other_objects"], e["false_tracks_wide_box"], e["false_tracks_on_background"]))
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("đã ghi", out)


if __name__ == "__main__":
    main()
