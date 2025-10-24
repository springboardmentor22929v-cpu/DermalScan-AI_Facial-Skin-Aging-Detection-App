# scripts/preprocess.py
import os, argparse, cv2, numpy as np
from utils import list_images_in_dir, read_image, save_image
from pathlib import Path
import hashlib

def apply_clahe_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return img2

def gamma_correction(img_rgb, gamma=1.0):
    invGamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** invGamma * 255
    table = np.clip(table,0,255).astype("uint8")
    return cv2.LUT(img_rgb, table)

def skin_mask_from_hsv(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    # broad skin range (tweak for your dataset)
    lower = np.array([0, 15, 40])
    upper = np.array([25, 200, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def crop_to_largest_mask_region(img_rgb, mask, pad=10):
    # find contours and crop bounding box of largest connected region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb  # fallback
    areas = [cv2.contourArea(c) for c in contours]
    idx = np.argmax(areas)
    x,y,w,h = cv2.boundingRect(contours[idx])
    x0 = max(0, x-pad); y0 = max(0, y-pad)
    x1 = min(img_rgb.shape[1], x+w+pad); y1 = min(img_rgb.shape[0], y+h+pad)
    return img_rgb[y0:y1, x0:x1]

def make_square_and_resize(img_rgb, size=224, pad_color=(0,0,0)):
    h, w = img_rgb.shape[:2]

    # Determine padding
    if h == w:
        sq = img_rgb
    else:
        diff = abs(h - w)
        if h > w:
            pad_left = diff // 2
            pad_right = diff - pad_left
            sq = cv2.copyMakeBorder(img_rgb, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
        else:
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            sq = cv2.copyMakeBorder(img_rgb, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=pad_color)

    # Resize to target size
    resized = cv2.resize(sq, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized

def normalize_imagenet_like(img_rgb, mean=0.5, std=0.5):
    # maps 0-255 to -1..1 if mean=0.5,std=0.5
    arr = img_rgb.astype('float32')/255.0
    arr = (arr - mean) / std
    return arr

def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def preprocess_image(path, settings):
    img = read_image(path)

    # 3. color/illum corrections
    if settings['clahe']:
        img = apply_clahe_rgb(img)

    # 7. Gamma Correction
    if settings['gamma'] != 1.0:
        img = gamma_correction(img, settings['gamma'])
    
    # 4. Skin segmentation and crop
    mask = skin_mask_from_hsv(img)
    img_cropped = crop_to_largest_mask_region(img, mask)

    # Resize
    img_final = make_square_and_resize(img_cropped, settings['size'])

    # # normalize for CNN 
    # # Normalization is for feeding the model, not for displaying images.
    # img_final = normalize_imagenet_like(img_final)
    
    return img_final
    

def main(input_dir, out_dir, size=224, clahe=True, gamma=1.0):
    pairs = list_images_in_dir(input_dir)
    os.makedirs(out_dir, exist_ok=True)
    seen_hashes = set()
    for src, cls in pairs:
        try:
            img_proc = preprocess_image(src, {'clahe':clahe,'gamma':gamma,'size':size})
        except Exception as e:
            print("Skipping corrupt or unreadable:", src, e)
            continue
        # deduplicate simple file-hash check of original
        h = file_hash(src)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        # save
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        fname = Path(src).stem + ".png"
        save_image(os.path.join(cls_dir, fname), img_proc)
    print("Preprocessing complete. Saved to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_raw")
    parser.add_argument("--out", default="cleaned_data")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--no-clahe", dest="clahe", action='store_false')
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    main(args.input, args.out, args.size, args.clahe, args.gamma)
