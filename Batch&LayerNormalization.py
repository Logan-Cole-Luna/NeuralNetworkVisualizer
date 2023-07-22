import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Sample data generation
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Adding a bias term to the input feature X
X_b = np.c_[np.ones((100, 1)), X]

# Model with Batch Normalization
model_with_bn = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(2,)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),  # Batch Normalization layer
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1)
])

# Model with Layer Normalization
model_with_ln = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(2,)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.LayerNormalization(),  # Layer Normalization layer
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_with_bn.compile(optimizer='adam', loss='mean_squared_error')

# Training with Batch Normalization
history_with_bn = model_with_bn.fit(X_b, y, epochs=100, verbose=0)

# Compile the model
model_with_ln.compile(optimizer='adam', loss='mean_squared_error')

# Training with Layer Normalization
history_with_ln = model_with_ln.fit(X_b, y, epochs=100, verbose=0)

# Plotting the loss curve
plt.title('Loss Convergence with Layer & Batch Normalization')
plt.plot(history_with_ln.history['loss'], label='Layer Normalization', color='orange')
plt.plot(history_with_bn.history['loss'], label='Batch Normalization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.legend()
plt.show()