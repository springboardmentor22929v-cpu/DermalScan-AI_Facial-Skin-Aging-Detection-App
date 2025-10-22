
"""
01_face_detection_and_crop.py
Walks raw_datasets/<class> and detects faces using MTCNN (preferred).
Crops face region, converts to RGB, optionally resizes (keeps aspect ratio) and writes to processed/<class>.

Usage:
 python 01_face_detection_and_crop.py --src "C:/Users/DELL/Aspire-Infolabs/raw_datasets" --dst "C:/Users/DELL/Aspire-Infolabs/processed" --min_size 224 --max_size 1024
"""
import argparse
from mtcnn import MTCNN
import cv2
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

# Haar cascade fallback
HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def detect_faces_mtcnn(detector, img_rgb):
    try:
        results = detector.detect_faces(img_rgb)
        return results
    except Exception:
        return []

def detect_faces_haar(haar, img_gray):
    rects = haar.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    res = []
    for (x,y,w,h) in rects:
        res.append({"box":[int(x),int(y),int(w),int(h)], "confidence": None})
    return res

def smart_crop_and_save(img_bgr, box, out_path, min_size=224, max_size=1024):
    x,y,w,h = box
    h_im, w_im = img_bgr.shape[:2]

    # expand box by 20% (but keep inside image)
    pad_w = int(0.2 * w)
    pad_h = int(0.2 * h)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_im, x + w + pad_w)
    y2 = min(h_im, y + h + pad_h)

    face = img_bgr[y1:y2, x1:x2].copy()
    # enforce min size by scaling up
    fh, fw = face.shape[:2]
    scale = 1.0
    if min(fh, fw) < min_size:
        scale = min_size / min(fh, fw)
    if max(fh, fw) * scale > max_size:
        scale = min(scale, max_size / max(fh, fw))
    if scale != 1.0:
        new_w = int(fw * scale)
        new_h = int(fh * scale)
        face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # convert to RGB for saving pipeline expectation; store as BGR because cv2.imwrite expects BGR
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), face)
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="raw_datasets root")
    p.add_argument("--dst", required=True, help="processed output root")
    p.add_argument("--min_size", type=int, default=224)
    p.add_argument("--max_size", type=int, default=1024)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit("src does not exist")

    detector = MTCNN()
    haar = cv2.CascadeClassifier(HAAR_XML)

    classes = [p for p in src.iterdir() if p.is_dir()]
    for cls in classes:
        in_files = list(cls.glob("*.*"))
        out_cls = dst / cls.name
        out_cls.mkdir(parents=True, exist_ok=True)
        for pth in tqdm(in_files, desc=f"Processing {cls.name}"):
            try:
                img = cv2.imdecode(np.fromfile(str(pth), dtype=np.uint8), cv2.IMREAD_COLOR)
                # fallback to cv2.imread if above fails
                if img is None:
                    img = cv2.imread(str(pth))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detect_faces_mtcnn(detector, img_rgb)
                if not faces:
                    # fallback to Haar
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detect_faces_haar(haar, gray)
                if not faces:
                    if args.debug:
                        print(f"No faces for {pth}")
                    continue
                # take the highest-confidence face (mtcnn returns confidence)
                faces_sorted = sorted(faces, key=lambda x: x.get("confidence") or 1.0, reverse=True)
                box = faces_sorted[0]["box"]
                # some boxes can be negative; fix
                x,y,w,h = box
                x = max(0,x); y = max(0,y); w = max(1,w); h = max(1,h)
                box = [x,y,w,h]
                outfn = out_cls / (pth.stem + ".jpg")
                smart_crop_and_save(img, box, outfn, min_size=args.min_size, max_size=args.max_size)
            except Exception as e:
                if args.debug:
                    print(f"Error {pth}: {e}")

    print("Done. Check processed/ per-class folders for cropped faces.")

if __name__ == "__main__":
    main()
