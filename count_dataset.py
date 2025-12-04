# File: count_dataset.py

import os
from collections import defaultdict

# -----------------------------
# 1️⃣  CONFIGURATION
# -----------------------------
CLEAN_FOLDER = "data_corrected_faces"     # your cleaned dataset
AUG_FOLDER = "data_augmented"             # <-- change this if different

VALID_EXT = {".jpg", ".jpeg", ".png"}

# -----------------------------
# 2️⃣  COUNT IMAGES IN A FOLDER
# -----------------------------
def count_images(folder):
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        return {}, 0

    class_counts = defaultdict(int)
    total = 0

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if os.path.splitext(file)[1].lower() in VALID_EXT:
                class_counts[class_name] += 1
                total += 1

    return class_counts, total


# -----------------------------
# 3️⃣  PRINT SUMMARY
# -----------------------------
def print_summary(title, counts, total):
    print("\n-------------------------------------------------")
    print(f"📌 {title}")
    print("-------------------------------------------------")
    print(f"Total Images: {total}\n")

    if not counts:
        print("No images found.\n")
        return

    for c, n in counts.items():
        print(f"  {c:<20} : {n}")

    print("\n")


# -----------------------------
# 4️⃣  TRAIN / VAL / TEST SPLIT
# -----------------------------
def calculate_split(total):
    train = int(total * 0.7)
    val = int(total * 0.2)
    test = total - train - val
    return train, val, test


# -----------------------------
# 5️⃣  MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":

    # --- Cleaned data ---
    cleaned_counts, cleaned_total = count_images(CLEAN_FOLDER)
    print_summary("CLEANED DATASET SUMMARY", cleaned_counts, cleaned_total)

    # --- Augmented data ---
    aug_counts, aug_total = count_images(AUG_FOLDER)
    print_summary("AUGMENTED DATASET SUMMARY", aug_counts, aug_total)

    # --- Split (cleaned dataset only) ---
    print("-------------------------------------------------")
    print("📌 TRAIN / VALIDATION / TEST SPLIT (Based on cleaned dataset)")
    print("-------------------------------------------------")

    train, val, test = calculate_split(cleaned_total)
    print(f"  Train:      {train}")
    print(f"  Validation: {val}")
    print(f"  Test:       {test}")
    print("-------------------------------------------------\n")

    print("✅ Counting complete!")
