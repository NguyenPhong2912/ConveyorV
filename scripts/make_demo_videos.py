"""Tạo video demo từ ảnh thật của TẬP TEST (mô hình chưa từng thấy khi huấn luyện).

- demo_conveyor_testset.mp4: các frame băng chuyền thật (đoạn cuối video gốc, thuộc tập test) theo thứ tự thời gian.
- demo_bottles_testset.mp4: ảnh chai thật pass/defect của tập test, mỗi ảnh giữ ~1.2 giây.

Sử dụng:
    python scripts/make_demo_videos.py            # đọc data/split_manifest.csv, ghi vào raw_videos/
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720


def letterbox(img: np.ndarray) -> np.ndarray:
    scale = min(W / img.shape[1], H / img.shape[0])
    resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), 32, np.uint8)
    y, x = (H - resized.shape[0]) // 2, (W - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def write_video(path: Path, frames: list, fps: float) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"[OK] {path} — {len(frames)} frame, {len(frames) / fps:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data", help="Thư mục dataset đã chia (có split_manifest.csv)")
    parser.add_argument("--out", default="raw_videos")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(data / "split_manifest.csv", encoding="utf-8")) if r["split"] == "test"]

    # 1) Frame băng chuyền theo thứ tự thời gian
    conveyor = sorted(
        (r for r in rows if r["source"] == "conveyor"),
        key=lambda r: int(re.search(r"(\d+)_\w+\.rf\.", r["file"]).group(1)),
    )
    frames = [letterbox(cv2.imread(str(data / "images" / "test" / r["file"]))) for r in conveyor]
    write_video(out / "demo_conveyor_testset.mp4", frames, fps=6)

    # 2) Ảnh chai pass/defect, xen kẽ để demo đủ cả hai lớp
    photos = [r for r in rows if r["source"] != "conveyor" and (int(r["n_pass"]) + int(r["n_defect"])) > 0]
    defect = [r for r in photos if int(r["n_defect"]) > 0]
    passed = [r for r in photos if int(r["n_defect"]) == 0]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(defect)
    rng.shuffle(passed)
    order = [r for pair in zip(passed, defect) for r in pair][:60]
    hold = 15  # 1.2s ở 12.5 fps
    frames = []
    for r in order:
        img = letterbox(cv2.imread(str(data / "images" / "test" / r["file"])))
        frames.extend([img] * hold)
    write_video(out / "demo_bottles_testset.mp4", frames, fps=12.5)


if __name__ == "__main__":
    main()
