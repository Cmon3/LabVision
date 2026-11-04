"""Calibrate Lab contrast threshold for /api/correct guard.
GT boxes (real colonies) must pass, random empty-agar boxes must fail.
Prints distribution and both rates.
"""
import os, json, pathlib, random, hashlib
from PIL import Image
import numpy as np

# load key
try:
    os.environ['META_API_KEY'] = open('.env').read().split('META_API_KEY=')[1].split('\n')[0].strip().strip('"').strip("'")
except Exception:
    pass

from labvision.segment import segment_point
from labvision.contrast import contrast_delta_e

def _overlaps(x1,y1,x2,y2, labels):
    for lb in labels:
        lx1,ly1,lx2,ly2 = lb["x"], lb["y"], lb["x"]+lb["width"], lb["y"]+lb["height"]
        ix1=max(x1,lx1); iy1=max(y1,ly1); ix2=min(x2,lx2); iy2=min(y2,ly2)
        if ix2>ix1 and iy2>iy1:
            return True
    return False

# pick 3 images with GT
roots = [
    pathlib.Path("data/agar_demo/AGAR_representative/lower-resolution/14684.jpg"),
    pathlib.Path("data/agar_demo/AGAR_representative/lower-resolution/13938.jpg"),
]
gt_deltas=[]
empty_deltas=[]
random.seed(0)
for img_path in roots:
    if not img_path.exists():
        continue
    j=json.loads((img_path.with_suffix(".json")).read_text())
    labels=j["labels"]
    data=img_path.read_bytes()
    arr=np.array(Image.open(img_path).convert("RGB"))
    print(f"\n== {img_path.name} GT={len(labels)}")
    # sample up to 5 GT
    for lb in labels[:5]:
        x1,y1,w,h = lb["x"], lb["y"], lb["width"], lb["height"]
        x2,y2=x1+w,y1+h
        # ensure box at least 20
        seg=segment_point(data, x1,y1,x2,y2)
        if not seg.boxes:
            print(f" GT {x1},{y1} no seg")
            continue
        masks=seg.decoded_masks()
        if masks.shape[0]==0:
            print(f" GT {x1},{y1} no mask")
            continue
        d=contrast_delta_e(arr, masks[0], None)
        if d is not None:
            gt_deltas.append(d)
            print(f" GT {x1},{y1},{x2},{y2} dE={d:.2f}")
    # sample 5 empty random not overlapping
    cnt=0; tries=0
    while cnt<5 and tries<40:
        tries+=1
        # random inside central 80% of image to stay inside plate
        x1=random.randint(200, 1800)
        y1=random.randint(200, 1800)
        w=random.randint(40,100); h=random.randint(40,100)
        x2=x1+w; y2=y1+h
        if _overlaps(x1,y1,x2,y2, labels):
            continue
        seg=segment_point(data, x1,y1,x2,y2)
        if not seg.boxes:
            continue
        masks=seg.decoded_masks()
        if masks.shape[0]==0:
            continue
        d=contrast_delta_e(arr, masks[0], None)
        if d is not None:
            empty_deltas.append(d)
            print(f" empty {x1},{y1},{x2},{y2} dE={d:.2f}")
            cnt+=1

print("\n=== SUMMARY ===")
print(f"GT deltas n={len(gt_deltas)}: min {min(gt_deltas):.2f} max {max(gt_deltas):.2f} mean {np.mean(gt_deltas):.2f} median {np.median(gt_deltas):.2f}")
print(f"empty deltas n={len(empty_deltas)}: min {min(empty_deltas):.2f} max {max(empty_deltas):.2f} mean {np.mean(empty_deltas):.2f} median {np.median(empty_deltas):.2f}")
# evaluate thresholds
for thr in [3,5,7,10,12,15,18,20]:
    tp = sum(1 for d in gt_deltas if d>=thr)/len(gt_deltas)*100 if gt_deltas else 0
    tn = sum(1 for d in empty_deltas if d< thr)/len(empty_deltas)*100 if empty_deltas else 0
    print(f"thr {thr:4.1f} GT-pass {tp:5.1f}% empty-reject {tn:5.1f}%")
# sorted
print("\nGT sorted", sorted([round(x,1) for x in gt_deltas])[:20])
print("empty sorted", sorted([round(x,1) for x in empty_deltas])[:20])
