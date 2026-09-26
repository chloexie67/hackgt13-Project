"""Combine labelled YOLO exports into one training folder for Colab.

* Every source must already use the model's class order (0 = person, 1 = ball); run
  remap_classes.py first on exports where Roboflow reordered them.
* --unstretch W H resizes a source's images back to their true shape. Roboflow's "Resize"
  preprocessing had squashed the broadcast frames to 1280x1280; YOLO labels are fractions of
  the image size, so they stay correct after resizing.
* Each source's train/valid/test splits are kept apart (test stays a held-out set).

Usage:
  python build_training_set.py OUT SRC[:WxH] [SRC[:WxH] ...]
  python build_training_set.py training_set labeled/broadcast_v1:1280x720 labeled/tactical_part_ready
Output: OUT/ (+ OUT.zip) with train/, valid/, test/ and data.yaml.
"""
import shutil
import sys
from pathlib import Path

import cv2

out = Path(sys.argv[1])
if out.exists():
    shutil.rmtree(out)
counts = {}
for spec in sys.argv[2:]:
    src, _, size = spec.partition(":")
    src = Path(src)
    size = tuple(int(v) for v in size.split("x")) if size else None
    for split_in in ("train", "valid", "val", "test"):
        imgs = sorted((src / split_in / "images").glob("*.jpg"))
        if not imgs:
            continue
        split = "valid" if split_in == "val" else split_in
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in imgs:
            name = f"{src.name}__{img.name}"
            if size:
                cv2.imwrite(str(out / split / "images" / name),
                            cv2.resize(cv2.imread(str(img)), size, interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy2(img, out / split / "images" / name)
            lab = src / split_in / "labels" / f"{img.stem}.txt"
            (out / split / "labels" / f"{Path(name).stem}.txt").write_text(
                lab.read_text() if lab.exists() else "")
        counts[(src.name, split)] = len(imgs)

splits = sorted({s for _, s in counts})
(out / "data.yaml").write_text(
    "path: /content/training_set\n" + "".join(f"{'val' if s == 'valid' else s}: {s}/images\n" for s in splits)
    + "names:\n  0: person\n  1: ball\n")
shutil.make_archive(str(out), "zip", out.parent, out.name)
for (src, split), n in sorted(counts.items()):
    print(f"{src:28s} {split:6s} {n:4d} images")
print(f"-> {out}/ and {out}.zip")
