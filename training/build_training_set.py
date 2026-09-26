"""Merge labelled YOLO exports into one training folder.

Sources must use the model's class order (run training.remap_classes first). SRC:WxH resizes
a source's images back to their true shape, for exports Roboflow squashed to squares; labels
are fractions of the image, so they stay valid.

Usage: python -m training.build_training_set OUT SRC[:WxH] [SRC[:WxH] ...]
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
