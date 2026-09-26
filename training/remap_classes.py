"""Rewrite a YOLO dataset's class ids into the given name order. Roboflow exports classes
alphabetically (ball=0, person=1); the model uses person=0, ball=1.

Usage: python -m training.remap_classes SRC_DIR DST_DIR person ball
"""
import re
import shutil
import sys
from pathlib import Path

src, dst, order = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
yaml = (src / "data.yaml").read_text()
names = re.search(r"names:\s*\[(.*?)\]", yaml)
names = [n.strip().strip("'\"") for n in names.group(1).split(",")] if names else \
    [m for m in re.findall(r"^\s+\d+:\s*(\S+)", yaml, re.M)]
mapping = {str(i): str(order.index(n)) for i, n in enumerate(names)}
print("source names", names, "-> new ids", mapping)

if dst.exists():
    shutil.rmtree(dst)
splits = []
for split in ("train", "valid", "val", "test"):
    if not (src / split / "images").exists():
        continue
    splits.append(split)
    shutil.copytree(src / split / "images", dst / split / "images")
    (dst / split / "labels").mkdir(parents=True)
    for lab in (src / split / "labels").glob("*.txt"):
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        (dst / split / "labels" / lab.name).write_text(
            "".join(" ".join([mapping[r[0]]] + r[1:]) + "\n" for r in rows))
val = "valid" if "valid" in splits else "val"
(dst / "data.yaml").write_text(f"path: .\ntrain: train/images\nval: {val}/images\nnames:\n"
                               + "".join(f"  {i}: {n}\n" for i, n in enumerate(order)))
print("wrote", dst)
