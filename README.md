# Face Aging Dataset Preparation

This repository contains scripts and processed data for preparing a facial aging dataset for deep learning. The dataset focuses on detecting aging signs like wrinkles, dark spots, puffy eyes, and clear skin. All preprocessing, augmentation, and label encoding steps have been included to create a clean and balanced dataset suitable for training a CNN.

---

## **Dataset**

- **Source:** [Kaggle Ageing Dataset - dark spots, puffy eyes, wrinkles](https://www.kaggle.com/datasets/mohit335448/ageing-dataset/data), [Multiple sources - clear skin](https://drive.google.com/drive/folders/13UxaFtRCCoOzMEt_f8ZjDJxnIKerP5Zb?usp=drive_link), [Everything is merged and put it google drive](https://drive.google.com/drive/folders/13UxaFtRCCoOzMEt_f8ZjDJxnIKerP5Zb?usp=drive_link)
- **Classes:**
    - `clear skin`
    - `dark spots`
    - `puffy eyes`
    - `wrinkles`
- **Initial Structure:**

data_raw/
    ageing_dataset/
        DATASET/
            clear skin/
            dark spots/
            puffy eyes/
            wrinkles/


- **Verification:**
    - Class distribution plots were generated to ensure balanced data.
    - Checked for missing or corrupted files.

---

## **Data Cleaning**

### **Steps Performed:**

1. **Color and Illumination Correction**
    - Applied CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve skin contrast.
    - Applied Gamma Correction to normalize brightness and enhance subtle features.

2. **Skin Segmentation and Cropping**
    - Converted images to HSV color space for skin detection.
    - Applied thresholding and morphological operations (dilation, closing) to clean masks.
    - Cropped the image to the **largest skin region**, removing background, hair, or clothing.

3. **Feature Enhancement**
    - Applied Gamma Correction to enhance the visibility of subtle aging signs.

**Output:** Cleaned images stored in `cleaned_data/` with class-wise folders.

---

## **Preprocessing**

- **Resizing:**
    - All images resized to **224×224 pixels**, with **white padding** to maintain aspect ratio.

- **Normalization:**
    - Pixel values scaled to a consistent range suitable for CNNs (0–1 or normalized per channel).

### Preprocess images: color correction, skin segmentation, resizing
```
!python scripts/preprocess.py --input data_raw/ageing_dataset/DATASET --out cleaned_data
```


---

## **Data Augmentation**

- **Deterministic augmentations applied to all images:**
    1. **Horizontal Flip**
    2. **Rotation** (±15°)
    3. **Zoom/Scale** (±15%)

- **Purpose:**
    - Increase dataset size and improve model generalization.
    - Each image gets three augmented versions with descriptive filenames, e.g., `_flip`, `_rot`, `_zoom`.

- **Output:** Augmented dataset stored in `augmented_data/`.
```
!python scripts/augment.py --clean cleaned_data --out augmented_data
```


---

## **One-Hot Encoding**

- **Class-to-index mapping:**
{'clear skin': 0, 'dark spots': 1, 'puffy eyes': 2, 'wrinkles': 3}

### Convert class labels to one-hot vectors
```
!python scripts/one_hot_encoding.py
```

