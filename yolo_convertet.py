import json, os
from collections import defaultdict

# mapping COCO category_id -> index keypoint (0..2)
# У тебя в JSON: origin=8, x_end=9, y_end=10
KPT_MAP = {8: 0, 9: 1, 10: 2}

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def coco_to_yolo_pose(coco_json_path, images_dir, out_labels_dir, pad_px=20):
    os.makedirs(out_labels_dir, exist_ok=True)
    coco = json.load(open(coco_json_path, "r", encoding="utf-8"))

    # image_id -> info
    images = {img["id"]: img for img in coco["images"]}

    # collect points per image: image_id -> list of 3 points (x,y) or None
    pts = defaultdict(lambda: [None, None, None])

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cid = ann["category_id"]
        if cid not in KPT_MAP:
            continue
        k = KPT_MAP[cid]
        x, y, w, h = ann["bbox"]
        # твой bbox 1x1: используем (x,y) как точку
        pts[img_id][k] = (float(x), float(y))

    # write one label per image
    for img_id, img in images.items():
        W, H = img["width"], img["height"]
        file_name = img["file_name"]
        stem = os.path.splitext(os.path.basename(file_name))[0]
        label_path = os.path.join(out_labels_dir, stem + ".txt")

        kpts = pts.get(img_id, [None, None, None])

        # если каких-то точек нет — лучше пропустить картинку (или поставить v=0)
        if any(p is None for p in kpts):
            # вариант: пропустить, чтобы не обучать на мусоре
            continue

        # bbox: по min/max keypoints + паддинг
        xs = [p[0] for p in kpts]
        ys = [p[1] for p in kpts]
        x0 = clip(min(xs) - pad_px, 0, W - 1)
        y0 = clip(min(ys) - pad_px, 0, H - 1)
        x1 = clip(max(xs) + pad_px, 0, W - 1)
        y1 = clip(max(ys) + pad_px, 0, H - 1)

        bw = max(2.0, x1 - x0)
        bh = max(2.0, y1 - y0)
        xc = x0 + bw / 2.0
        yc = y0 + bh / 2.0

        # normalize
        xc_n, yc_n = xc / W, yc / H
        bw_n, bh_n = bw / W, bh / H

        # keypoints normalize + visibility=2
        kpt_flat = []
        for (x, y) in kpts:
            kpt_flat += [x / W, y / H, 2]

        # one object "chart" => class 0
        line = "0 " + " ".join(f"{v:.6f}" for v in [xc_n, yc_n, bw_n, bh_n] + kpt_flat)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write(line + "\n")

            
if __name__ == "__main__":
    base = "dataset_yolo"

    coco_to_yolo_pose(
        coco_json_path=os.path.join(base, "annotations", "train.json"),
        images_dir=os.path.join(base, "images", "train"),
        out_labels_dir=os.path.join(base, "labels", "train"),
        pad_px=20
    )

    coco_to_yolo_pose(
        coco_json_path=os.path.join(base, "annotations", "val.json"),
        images_dir=os.path.join(base, "images", "val"),
        out_labels_dir=os.path.join(base, "labels", "val"),
        pad_px=20
    )
