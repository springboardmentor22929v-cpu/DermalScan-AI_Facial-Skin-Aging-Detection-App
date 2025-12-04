# File: scripts/evaluate_model.py 
 
import tensorflow as tf 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from tqdm import tqdm 
 
# Import the dataset variables directly from prepare_dataset.py 
from prepare_dataset import test_ds, class_names, IMG_SIZE, BATCH_SIZE 
 
# --- Configuration --- 
MODELS_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\models" 
EVALUATION_OUTPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\evaluation_results" 
 
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True) 
 
# Path to your best saved model (from train_model.py's ModelCheckpoint) 
MODEL_PATH = os.path.join(MODELS_DIR, 'efficientnetb0_skin_model_best.h5') 
 
# --- Load Model --- 
print(f"Loading model from: {MODEL_PATH}") 
try: 
    model = tf.keras.models.load_model(MODEL_PATH) 
    print("Model loaded successfully!") 
except Exception as e: 
    print(f"Error loading model: {e}") 
    print("Please ensure the model path is correct and the model has been trained.") 
    exit() 
 
# --- Evaluate on Test Set --- 
print("\nEvaluating model on the test dataset...") 
test_loss, test_accuracy = model.evaluate(test_ds) 
print(f"Test Loss: {test_loss:.4f}") 
print(f"Test Accuracy: {test_accuracy:.4f}") 
 
# --- Generate Predictions for Detailed Metrics --- 
print("\nGenerating predictions for detailed metrics...") 
y_true_labels = [] # Stores actual integer labels 
y_pred_classes_full = [] # Stores predicted integer class indices for the full test set 
 
# Get all true labels and predictions from the test_ds 
# We need to collect all predictions first for classification_report and confusion_matrix 
for images_batch, labels_batch in tqdm(test_ds, desc="Predicting on test set"): 
    preds_batch = model.predict(images_batch, verbose=0) 
    y_true_labels.extend(np.argmax(labels_batch.numpy(), axis=1)) # Convert one-hot labels back to integer for comparison 
    y_pred_classes_full.extend(np.argmax(preds_batch, axis=1)) 
 
y_true_labels = np.array(y_true_labels) 
y_pred_classes_full = np.array(y_pred_classes_full) 
 
# --- Classification Report --- 
print("\n--- Classification Report ---") 
report = classification_report(y_true_labels, y_pred_classes_full, target_names=class_names) 
print(report) 
report_path = os.path.join(EVALUATION_OUTPUT_DIR, 'classification_report.txt') 
with open(report_path, 'w') as f: 
    f.write(report) 
print(f"Classification report saved to: {report_path}") 
 
# --- Confusion Matrix --- 
print("\n--- Confusion Matrix ---") 
cm = confusion_matrix(y_true_labels, y_pred_classes_full) 
plt.figure(figsize=(len(class_names)+2, len(class_names)+2)) # Adjust figure size dynamically 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False) 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.title('Confusion Matrix') 
plt.tight_layout() 
confusion_matrix_path = os.path.join(EVALUATION_OUTPUT_DIR, 'confusion_matrix.png') 
plt.savefig(confusion_matrix_path) 
print(f"Confusion matrix saved to: {confusion_matrix_path}") 
plt.show() # Display the plot 
 
 
# --- Visualize some predictions --- 
print("\n--- Visualizing Sample Predictions ---") 
 
# Collect a few batches from the test_ds for visualization 
images_for_viz = [] 
labels_for_viz = [] 
predictions_for_viz = [] 
num_batches_to_collect = 1 # Take just one batch for visualization (BATCH_SIZE * 1 images) 
 
# Iterate through the test_ds to get images, labels, and predictions for visualization 
for images_batch, labels_batch in test_ds.take(num_batches_to_collect): 
    preds_batch = model.predict(images_batch, verbose=0) 
    images_for_viz.append(images_batch.numpy()) 
    labels_for_viz.append(labels_batch.numpy()) # Labels are already one-hot 
    predictions_for_viz.append(preds_batch) 
 
# Concatenate collected batches (if more than one) 
images_for_viz = np.concatenate(images_for_viz, axis=0) 
labels_for_viz = np.concatenate(labels_for_viz, axis=0) # These are still one-hot 
predictions_for_viz = np.concatenate(predictions_for_viz, axis=0) 
 
plt.figure(figsize=(12, 12)) 
num_samples_to_show = min(9, images_for_viz.shape[0]) # Show up to 9 samples from the collected batch 
 
for i in range(num_samples_to_show): 
    ax = plt.subplot(3, 3, i + 1) 
     
    # Undo normalization for display (images are already 0-1 from tf.data.Dataset) 
    display_image = images_for_viz[i] * 255 # Scale back to 0-255 
    display_image = display_image.astype(np.uint8) 
 
    predicted_class_idx = np.argmax(predictions_for_viz[i]) 
    confidence = np.max(predictions_for_viz[i]) 
     
    predicted_label = class_names[predicted_class_idx] 
    true_label_idx = np.argmax(labels_for_viz[i]) # Convert true one-hot label to integer index 
    true_label = class_names[true_label_idx] 
 
    plt.imshow(display_image) 
    color = "green" if predicted_label == true_label else "red" 
    plt.title(f"True: {true_label}\nPred: {predicted_label} ({confidence*100:.1f}%)", color=color) 
    plt.axis("off") 
plt.tight_layout() 
sample_predictions_path = os.path.join(EVALUATION_OUTPUT_DIR, 'sample_predictions.png') 
plt.savefig(sample_predictions_path) 
print(f"Sample predictions saved to: {sample_predictions_path}") 
plt.show() # Display the plot 
 
 
print("\nComprehensive model evaluation complete!")