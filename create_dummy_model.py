import tensorflow as tf

# Dummy model for 4 classes
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
])

# Save the model
model.save("your_model.h5")
print("Dummy model created!")
