# test_plot.py
import matplotlib.pyplot as plt
import numpy as np

# Simulated training history
epochs = 50
train_acc = np.linspace(0.5, 1.0, epochs) + np.random.uniform(-0.02, 0.02, epochs)
val_acc = np.linspace(0.4, 0.97, epochs) + np.random.uniform(-0.02, 0.02, epochs)
train_loss = np.linspace(1.0, 0.002, epochs) + np.random.uniform(-0.01, 0.01, epochs)
val_loss = np.linspace(1.2, 0.095, epochs) + np.random.uniform(-0.01, 0.01, epochs)

# Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# show_accuracy.py
# Simulated accuracy values based on your previous model training

train_accuracy = 1.0        # 100% training accuracy
val_accuracy = 0.9779       # 97.79% validation accuracy

print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
