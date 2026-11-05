"""HF mrm8488/ImageNet1K-val parquet → torchvision ImageNet 레이아웃(<dst>/meta.bin, <dst>/val/<wnid>/*.JPEG).
meta.bin = (wnid_to_classes, val_wnids): val_wnids는 val 파일명 정렬 순서와 맞춘다.
사용: python imagenet_from_parquet.py --src <parquet 폴더> --dst <root>
"""

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--src", required=True, type=Path)
parser.add_argument("--dst", required=True, type=Path)
args = parser.parse_args()

files = sorted(args.src.glob("data/*.parquet"))
if not files:
    raise SystemExit(f"parquet 없음: {args.src}/data")
val_dir = args.dst / "val"
val_dir.mkdir(parents=True, exist_ok=True)

info = json.loads((args.src / "dataset_infos.json").read_text(encoding="utf-8"))
wnids = next(iter(info.values()))["features"]["label"]["names"]
assert len(wnids) == 1000, len(wnids)
n = 0
for f in files:
    table = pq.read_table(f)
    for row in table.to_pylist():
        wnid = wnids[row["label"]]
        out = val_dir / wnid
        out.mkdir(exist_ok=True)
        # 원본 파일명은 없으므로 전역 순번으로 이름을 만든다(클래스 폴더 안 정렬 순서만 의미 있음).
        path = out / f"ILSVRC2012_val_{n:08d}.JPEG"
        if not path.exists():
            img = row["image"]
            data = img["bytes"] if isinstance(img, dict) else img
            Image.open(io.BytesIO(data)).convert("RGB").save(path, "JPEG", quality=95)
        n += 1
    print(f"{f.name}: 누적 {n}", flush=True)

# torchvision ImageNet: val_wnids는 val/ 아래 전체 파일을 정렬했을 때의 wnid 순서
all_files = sorted(p for p in val_dir.rglob("*.JPEG"))
val_wnids = [p.parent.name for p in all_files]
wnid_to_classes = {w: (w,) for w in wnids}
torch.save((wnid_to_classes, val_wnids), args.dst / "meta.bin")
print(
    f"완료: {len(all_files)}장, 클래스 {len(set(val_wnids))}, meta.bin 저장 → {args.dst}"
)
