import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def draw_keypoints(img, kpts):
    labels = ["origin", "x_end", "y_end"]
    colors = [(0, 255, 255), (0, 255, 0), (0, 128, 255)]

    for (x, y), label, color in zip(kpts, labels, colors):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(img, pt, 5, color, -1)
        cv2.putText(
            img,
            label,
            (pt[0] + 6, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    origin = (int(round(kpts[0][0])), int(round(kpts[0][1])))
    x_end = (int(round(kpts[1][0])), int(round(kpts[1][1])))
    y_end = (int(round(kpts[2][0])), int(round(kpts[2][1])))
    cv2.line(img, origin, x_end, (0, 200, 0), 2)
    cv2.line(img, origin, y_end, (0, 128, 255), 2)


def process_image(img_path, model, out_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skip (cannot read): {img_path}")
        return

    results = model(img)
    if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        print(f"Skip (no keypoints): {img_path}")
        return

    kpts = results[0].keypoints.xy[0].cpu().numpy()
    if kpts.shape[0] < 3:
        print(f"Skip (expected 3 points): {img_path}")
        return

    draw_keypoints(img, kpts[:3])

    out_path = out_dir / f"{img_path.stem}_result.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")


def iter_images(input_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO pose model and save images with 3 keypoints marked."
    )
    parser.add_argument(
        "--model",
        default="runs/pose/train/weights/best.pt",
        help="Path to YOLO pose model weights.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a single image. If omitted, input directory is used.",
    )
    parser.add_argument(
        "--input-dir",
        default="images_in",
        help="Directory with images to process.",
    )
    parser.add_argument(
        "--output-dir",
        default="images_out",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        process_image(Path(args.image), model, out_dir)
        return

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for img_path in iter_images(input_dir):
        process_image(img_path, model, out_dir)


if __name__ == "__main__":
    main()
