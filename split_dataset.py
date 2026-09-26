"""Split a pre-labelled dataset into equal parts for several labellers.

Frames are dealt out alternately (sorted by clip and time), so every part gets a mix of all
clips and of train/val frames. Each part is a complete YOLO dataset with its own data.yaml
and a zip ready to upload to Roboflow.

Usage: python split_dataset.py dataset_tactical 2
Output: dataset_tactical_part1/ (+ .zip), dataset_tactical_part2/ (+ .zip), ...
"""
import shutil
import sys
from pathlib import Path

src, parts = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2
yaml = (src / "data.yaml").read_text()
names_block = yaml[yaml.index("names:"):]

counts = [dict(train=0, val=0) for _ in range(parts)]
for split in ("train", "val"):
    images = sorted((src / split / "images").glob("*.jpg"))
    for i, img in enumerate(images):
        k = i % parts
        dst = Path(f"{src}_part{k + 1}")
        for sub, f in (("images", img), ("labels", src / split / "labels" / f"{img.stem}.txt")):
            (dst / split / sub).mkdir(parents=True, exist_ok=True)
            if f.exists():
                shutil.copy2(f, dst / split / sub / f.name)
        counts[k][split] += 1

for k in range(parts):
    dst = Path(f"{src}_part{k + 1}")
    (dst / "data.yaml").write_text(f"path: .\ntrain: train/images\nval: val/images\n{names_block}")
    shutil.make_archive(str(dst), "zip", dst)
    print(f"{dst}: {counts[k]['train']} train + {counts[k]['val']} val frames -> {dst}.zip")
