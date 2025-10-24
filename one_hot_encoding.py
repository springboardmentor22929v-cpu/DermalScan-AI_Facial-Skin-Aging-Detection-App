import os
import numpy as np

# Path to dataset
data_dir = "augmented_data/"

# Get class names sorted
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print("Classes:", classes)

# Map class names to indices
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
print("Class to index mapping:", class_to_idx)

# Example: collect images and their labels
image_paths = []
labels = []

for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    for img_file in os.listdir(cls_path):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(cls_path, img_file))
            labels.append(class_to_idx[cls])

# Convert labels to NumPy array
labels = np.array(labels)

# One-hot encode labels
num_classes = len(classes)
one_hot_labels = np.eye(num_classes)[labels]

print("Example one-hot label for first image:", one_hot_labels[1221])
