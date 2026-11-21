"""Thử SceneCutDetector của worker trên một video: so sánh phán quyết theo vết khi có và không có phát hiện chuyển cảnh.

Sử dụng:  python scripts/scenecut_check.py --model models/conveyor_yolo26_30ep_best.pt --video raw_videos/conveyor_demo.mp4
(báo cáo dùng checkpoint lần 1 conveyor_yolo26_30ep_best.pt và video demo conveyor_demo.mp4; muốn đo cho lần 2 thì đổi --model sang models/conveyor_qc_yolo26n.pt)
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
from ultralytics import YOLO
from conveyorvision.web.worker import SceneCutDetector, clear_tracker_state

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="raw_videos/conveyor_demo.mp4")
ap.add_argument("--model", default="models/conveyor_qc_yolo26n.pt")
ap.add_argument("--out", default="doc/evaluation/extra/scenecut_check.json")
args = ap.parse_args()
VIDEO, MODEL = args.video, args.model
conf_t, B, V = 0.45, 30, 3
out = {}
for imgsz in (640, 480):
    for use_cut in (False, True):
        model = YOLO(MODEL)
        det = SceneCutDetector()
        cap = cv2.VideoCapture(VIDEO)
        tr, k, cuts = {}, 0, []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            k += 1
            cut = False
            if use_cut:
                cut = det.update(fr)
                if cut:
                    cuts.append(k)
                    clear_tracker_state(model)
            r = model.track(fr, persist=True, tracker="bytetrack.yaml", conf=conf_t, imgsz=imgsz, verbose=False)[0]
            seen = set()
            if r.boxes is not None and r.boxes.id is not None:
                for tid, c, s in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    seen.add(tid)
                    t = tr.setdefault(tid, dict(first=k, last=k, p=0, d=0, dmax=0.0, state="INSPECTING", at=None))
                    t["last"] = k
                    if c == 1:
                        t["d"] += 1; t["dmax"] = max(t["dmax"], s)
                    else:
                        t["p"] += 1
                    if t["state"] == "INSPECTING" and (t["d"] >= 2 or (t["d"] >= 1 and s >= 0.5)):
                        t["state"], t["at"] = "DEFECT", k
            for tid, t in tr.items():
                if t["state"] == "INSPECTING":
                    lost = k - t["last"]
                    if lost >= B or (cut and lost > 0):
                        t["state"] = "PASS" if t["p"] + t["d"] >= V else "INVALID"
                        t["at"] = k
        cap.release()
        for t in tr.values():
            if t["state"] == "INSPECTING":
                t["state"] = "PASS" if t["p"] + t["d"] >= V else "INVALID"; t["at"] = "end"
        key = "%d_%s" % (imgsz, "cut" if use_cut else "nocut")
        out[key] = {"cuts": cuts, "tracks": {str(i): t for i, t in tr.items()}}
        cnt = {s: sum(1 for t in tr.values() if t["state"] == s) for s in ("PASS", "DEFECT", "INVALID")}
        print(key, "cuts:", cuts, "| tracks:", len(tr), cnt)
        for i, t in tr.items():
            print("   #%d %d-%d p/d=%d/%d -> %s @%s" % (i, t["first"], t["last"], t["p"], t["d"], t["state"], t["at"]))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
