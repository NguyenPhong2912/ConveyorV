"""Chuẩn hoá 3 dataset Roboflow về 2 lớp (0: pass, 1: defect) và chia train/val/test THEO NHÓM.

Vì sao không dùng split gốc của Roboflow:
- conveyor-belt là các frame cắt từ cùng một video → frame liền kề gần như giống hệt nhau;
- hai dataset ảnh web có thể chứa cùng một ảnh.
Chia ngẫu nhiên từng ảnh khiến ảnh gần trùng nằm ở cả train lẫn test → mAP bị thổi phồng.

Cách làm:
1. Nguồn video: chia theo THỜI GIAN — 70% đầu video → train, 15% kế → val, 15% cuối → test,
   bỏ VIDEO_GAP frame ở mỗi điểm cắt để không có frame liền kề nằm ở hai tập khác nhau.
2. Nguồn ảnh: gộp ảnh gần trùng (pHash 64-bit, Hamming ≤ NEAR_DUP_BITS) vào cùng nhóm, kể cả khác
   dataset, rồi gán nguyên nhóm vào train/val/test (70/15/15), phân tầng theo (nguồn, có defect không).

Sử dụng:
    python scripts/prepare_dataset.py                 # đọc raw_datasets/, ghi data/
    python scripts/prepare_dataset.py --raw raw_datasets --out data --seed 42
"""

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from conveyorvision.dataset_manager import DEFAULT_NAME_MAP  # noqa: E402

# (thư mục trong raw_datasets, tiền tố tên file, có phải frame cắt từ video không)
SOURCES = [
    ("conveyor-belt-eafpe-bgzzm", "conveyor", True),
    ("bottle-detection-qti1t", "bayo", False),
    ("bottle_defects-tpmcq-kijtp", "bottle_defect", False),
]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
NEAR_DUP_BITS = 6
VIDEO_GAP = 10
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash(path: Path) -> int:
    """Perceptual hash 64-bit (DCT 32x32 → 8x8 tần số thấp so với trung vị)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_to_all(hashes: np.ndarray, h: int) -> np.ndarray:
    xor = np.bitwise_xor(hashes, np.uint64(h))
    return np.array([bin(int(v)).count("1") for v in xor])


def class_map_for(source_dir: Path) -> dict:
    """Ánh xạ id lớp nguồn → 0/1 theo TÊN lớp (thứ tự lớp mỗi dataset mỗi khác)."""
    names = yaml.safe_load((source_dir / "data.yaml").read_text(encoding="utf-8"))["names"]
    items = names.items() if isinstance(names, dict) else enumerate(names)
    mapping = {}
    for idx, name in items:
        key = str(name).strip().lower()
        if key not in DEFAULT_NAME_MAP:
            raise SystemExit(f"Không có quy tắc map cho lớp '{name}' trong {source_dir.name}")
        mapping[int(idx)] = DEFAULT_NAME_MAP[key]
    return mapping


def read_labels(label_path: Path, mapping: dict) -> list:
    lines = []
    if label_path.exists():
        for raw in label_path.read_text().splitlines():
            parts = raw.split()
            if len(parts) == 5:  # bỏ nhãn polygon/segment, chỉ giữ bbox YOLO
                lines.append(f"{mapping[int(parts[0])]} {' '.join(parts[1:])}")
    return lines


def collect(raw_dir: Path) -> list:
    items = []
    for folder, prefix, is_video in SOURCES:
        src = raw_dir / folder
        if not src.exists():
            raise SystemExit(f"Thiếu dataset {src} — tải bằng dataset_manager.py hoặc notebook trước.")
        mapping = class_map_for(src)
        for img in sorted(src.glob("*/images/*")):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            label = img.parent.parent / "labels" / (img.stem + ".txt")
            frame = re.search(r"(\d+)_\w+\.rf\.", img.name) if is_video else None
            items.append({
                "path": img,
                "source": prefix,
                "is_video": is_video,
                "orig_split": img.parent.parent.name,
                "labels": read_labels(label, mapping),
                "hash": phash(img),
                "frame": int(frame.group(1)) if frame else None,
            })
    return items


def split_video(items: list) -> dict:
    """Chia frame video theo thời gian; frame nằm trong khe VIDEO_GAP quanh điểm cắt bị loại (None)."""
    split_of = {}
    video = sorted((i for i, it in enumerate(items) if it["is_video"]), key=lambda i: items[i]["frame"])
    if not video:
        return split_of
    first, last = items[video[0]]["frame"], items[video[-1]]["frame"]
    cut_val = first + SPLIT_RATIOS["train"] * (last - first)
    cut_test = cut_val + SPLIT_RATIOS["val"] * (last - first)
    for i in video:
        f = items[i]["frame"]
        if abs(f - cut_val) <= VIDEO_GAP / 2 or abs(f - cut_test) <= VIDEO_GAP / 2:
            split_of[i] = None
        else:
            split_of[i] = "train" if f < cut_val else "val" if f < cut_test else "test"
    return split_of


def group_photos(items: list) -> dict:
    """Union-find các ảnh (không phải video) gần trùng nhau → {chỉ số ảnh: mã nhóm}."""
    photos = [i for i, it in enumerate(items) if not it["is_video"]]
    parent = {i: i for i in photos}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = np.array([items[i]["hash"] for i in photos], dtype=np.uint64)
    for k, i in enumerate(photos):
        dist = hamming_to_all(hashes[k + 1:], items[i]["hash"])
        for j in np.nonzero(dist <= NEAR_DUP_BITS)[0]:
            parent[find(i)] = find(photos[k + 1 + int(j)])
    return {i: find(i) for i in photos}


def split_photos(items: list, group_of: dict, seed: int) -> dict:
    members = defaultdict(list)
    for i, g in group_of.items():
        members[g].append(i)

    # Phân tầng theo (nguồn chính của nhóm, nhóm có defect không)
    strata = defaultdict(list)
    for g, idx in members.items():
        source = Counter(items[i]["source"] for i in idx).most_common(1)[0][0]
        has_defect = any(l.startswith("1 ") for i in idx for l in items[i]["labels"])
        strata[(source, has_defect)].append(g)

    rng = random.Random(seed)
    split_of = {}
    for key in sorted(strata):
        group_ids = sorted(strata[key])
        rng.shuffle(group_ids)
        total = sum(len(members[g]) for g in group_ids)
        filled = Counter()
        for g in group_ids:
            # chọn split đang thiếu nhiều nhất so với tỉ lệ mục tiêu
            split = max(SPLIT_RATIOS, key=lambda s: SPLIT_RATIOS[s] * total - filled[s])
            filled[split] += len(members[g])
            for i in members[g]:
                split_of[i] = split
    return split_of


def cross_split_near_dups(items: list, split_of: dict) -> int:
    """Số ảnh val/test có ảnh gần trùng trong train (phải = 0 sau khi chia theo nhóm)."""
    train = np.array([it["hash"] for i, it in enumerate(items) if split_of.get(i) == "train"], dtype=np.uint64)
    return sum(
        1 for i, it in enumerate(items)
        if split_of.get(i) in ("val", "test") and hamming_to_all(train, it["hash"]).min() <= NEAR_DUP_BITS
    )


def write_output(items: list, group_of: dict, split_of: dict, out_dir: Path) -> None:
    for sub in ("images", "labels"):
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)
        for split in SPLIT_RATIOS:
            (out_dir / sub / split).mkdir(parents=True)

    with open(out_dir / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "source", "group", "split", "roboflow_split", "n_pass", "n_defect"])
        for i, it in enumerate(items):
            split = split_of[i]
            if split is None:
                continue
            name = f"{it['source']}_{it['path'].name}"
            shutil.copy2(it["path"], out_dir / "images" / split / name)
            (out_dir / "labels" / split / (Path(name).stem + ".txt")).write_text(
                "\n".join(it["labels"]) + ("\n" if it["labels"] else ""))
            n_def = sum(l.startswith("1 ") for l in it["labels"])
            writer.writerow([name, it["source"], group_of[i], split, it["orig_split"], len(it["labels"]) - n_def, n_def])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default="raw_datasets", help="Thư mục chứa 3 dataset Roboflow đã tải")
    parser.add_argument("--out", default="data", help="Thư mục đích (ghi đè data/images, data/labels)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = collect(Path(args.raw))
    photo_groups = group_photos(items)
    split_of = {**split_video(items), **split_photos(items, photo_groups, args.seed)}
    group_of = {**photo_groups, **{i: f"video_{it['frame']}" for i, it in enumerate(items) if it["is_video"]}}

    # So sánh rò rỉ: split gốc Roboflow vs split theo nhóm
    roboflow_split = {i: {"valid": "val"}.get(it["orig_split"], it["orig_split"]) for i, it in enumerate(items)}
    dropped = sum(1 for v in split_of.values() if v is None)
    leak_roboflow = cross_split_near_dups(items, roboflow_split)
    leak_grouped = cross_split_near_dups(items, split_of)
    print(f"Ảnh: {len(items)} | Nhóm ảnh tĩnh: {len(set(photo_groups.values()))} | Frame bị loại ở khe cắt video: {dropped}")
    print(f"Ảnh val/test có bản gần trùng trong train — split Roboflow: {leak_roboflow} | split theo nhóm: {leak_grouped}")

    out_dir = Path(args.out)
    write_output(items, group_of, split_of, out_dir)

    report = {
        "method": "Video: chia theo thời gian (70/15/15, bỏ khe cắt); ảnh tĩnh: nhóm gần trùng pHash rồi chia theo nhóm, phân tầng theo nguồn & defect",
        "near_dup_threshold_bits": NEAR_DUP_BITS,
        "total_images": len(items) - dropped,
        "dropped_video_gap_frames": dropped,
        "leak_roboflow_split": leak_roboflow,
        "leak_group_split": leak_grouped,
        "splits": {},
        "sources": {prefix: sum(1 for i, it in enumerate(items) if it["source"] == prefix and split_of[i])
                    for _, prefix, _ in SOURCES},
    }
    print(f"\n{'split':6s} {'ảnh':>6s} {'box pass':>9s} {'box defect':>11s}")
    for split in SPLIT_RATIOS:
        idx = [i for i in split_of if split_of[i] == split]
        boxes = Counter(l[0] for i in idx for l in items[i]["labels"])
        report["splits"][split] = {"images": len(idx), "pass_boxes": boxes["0"], "defect_boxes": boxes["1"]}
        print(f"{split:6s} {len(idx):6d} {boxes['0']:9d} {boxes['1']:11d}")
    (out_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã ghi {args.out}/images, {args.out}/labels, split_manifest.csv và split_report.json")


if __name__ == "__main__":
    main()
