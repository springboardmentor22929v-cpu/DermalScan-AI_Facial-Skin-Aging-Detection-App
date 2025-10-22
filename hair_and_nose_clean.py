#!/usr/bin/env python3
"""
02_hair_and_nose_clean.py
Performs hair/strand removal and nostril/dark-patch inpainting using morphological black-hat + threshold + cv2.inpaint.
Process images from processed/ and writes to cleaned/.

Usage:
 python 02_hair_and_nose_clean.py --src "C:/.../processed" --dst "C:/.../cleaned" --debug
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def remove_hair_and_inpaint(img_bgr, debug=False):
    # Convert to grayscale and enhance hair-like structures using black-hat
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # morphological blackhat to find dark strands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # threshold to binary mask
    _, mask_hair = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # refine mask: morphological closing then opening
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_hair = cv2.morphologyEx(mask_hair, cv2.MORPH_CLOSE, kernel2, iterations=1)
    mask_hair = cv2.morphologyEx(mask_hair, cv2.MORPH_OPEN, kernel2, iterations=1)
    # find strong dark spots (possible nostrils) by local thresholding in nose region
    # We'll use a heuristic: search central-lower region for dark blobs
    h, w = gray.shape
    y1 = int(h * 0.35)
    y2 = int(h * 0.9)
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    roi = gray[y1:y2, x1:x2]
    # adaptive threshold (inverted) to find dark areas
    thr = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    # clean small noise
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask_nose = np.zeros_like(gray)
    mask_nose[y1:y2, x1:x2] = thr

    # Combine masks
    mask = cv2.bitwise_or(mask_hair, mask_nose)
    # optionally dilate slightly to ensure coverage
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

    # For inpainting, we need 8-bit single channel mask with non-zero where to inpaint
    inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.subplot(1,3,1); plt.title("orig"); plt.axis('off'); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.subplot(1,3,2); plt.title("mask"); plt.axis('off'); plt.imshow(mask, cmap='gray')
        plt.subplot(1,3,3); plt.title("inpainted"); plt.axis('off'); plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
        plt.show()

    return inpainted

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="processed input root")
    p.add_argument("--dst", required=True, help="cleaned output root")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit("src does not exist")

    for cls in [d for d in src.iterdir() if d.is_dir()]:
        out_cls = dst / cls.name
        out_cls.mkdir(parents=True, exist_ok=True)
        files = list(cls.glob("*.*"))
        for f in tqdm(files, desc=f"Cleaning {cls.name}"):
            try:
                img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.imread(str(f))
                if img is None:
                    continue
                cleaned = remove_hair_and_inpaint(img, debug=args.debug)
                outp = out_cls / f.name
                # save using imwrite (works with unicode on Windows if path is str)
                cv2.imwrite(str(outp), cleaned)
            except Exception as e:
                if args.debug:
                    print(f"Failed {f}: {e}")

    print("Cleaning done. Check cleaned/")

if __name__ == "__main__":
    main()
