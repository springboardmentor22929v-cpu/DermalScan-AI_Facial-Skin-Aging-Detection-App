# scripts/utils.py
import os, cv2, numpy as np, pandas as pd
from PIL import Image

def list_images_in_dir(root):
    out = []
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.lower().endswith(('.jpg','.jpeg','.png','bmp','webp')):
                    out.append((os.path.join(p,f), cls))
    return out

def read_image(path, to_rgb=True):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {path}")
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image(path, arr):
    # use imencode + tofile to support unicode paths on Windows
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1]
    _, enc = cv2.imencode(ext, arr_bgr)
    enc.tofile(path)
