# scripts/augment.py
import os, argparse
from pathlib import Path
from utils import list_images_in_dir, read_image, save_image
import albumentations as A

def get_augmenters():
    """
    Returns individual augmenters
    - Horizontal flip
    - Rotation
    - Zoom (scale)
    """
    aug_flip = A.HorizontalFlip(p=1.0)      # always apply flip
    aug_rotate = A.Rotate(limit=15, p=1.0)  # rotate ±15°
    aug_zoom = A.ShiftScaleRotate(shift_limit=0, scale_limit=0.15, rotate_limit=0, p=1.0)  # zoom ±15%
    return aug_flip, aug_rotate, aug_zoom

def augment_all_images(clean_dir, out_dir):
    """
    Augment every image in clean_dir using flip, rotate, and zoom.
    Saves originals and augmented images to out_dir.
    """
    pairs = list_images_in_dir(clean_dir)
    class_map = {}
    for p, c in pairs:
        class_map.setdefault(c, []).append(p)
    
    aug_flip, aug_rotate, aug_zoom = get_augmenters()
    os.makedirs(out_dir, exist_ok=True)

    for cls, files in class_map.items():
        cls_out = os.path.join(out_dir, cls)
        os.makedirs(cls_out, exist_ok=True)

        for src in files:
            img = read_image(src)
            fname = Path(src).stem

            # save original
            save_image(os.path.join(cls_out, f"{fname}.png"), img)

            # flip
            img_flip = aug_flip(image=img)['image']
            save_image(os.path.join(cls_out, f"{fname}_flip.png"), img_flip)

            # rotate
            img_rotate = aug_rotate(image=img)['image']
            save_image(os.path.join(cls_out, f"{fname}_rot.png"), img_rotate)

            # zoom
            img_zoom = aug_zoom(image=img)['image']
            save_image(os.path.join(cls_out, f"{fname}_zoom.png"), img_zoom)

        print(f"{cls}: {len(files)} originals augmented with flip, rotate, zoom")

    print("Augmentation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="cleaned_data", help="Path to preprocessed dataset")
    parser.add_argument("--out", default="augmented_data", help="Path to save augmented images")
    args = parser.parse_args()

    augment_all_images(args.clean, args.out)
