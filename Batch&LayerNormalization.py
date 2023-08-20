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


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Sample data generation
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Adding a bias term to the input feature X
X_b_tensor = torch.cat((torch.ones((100, 1)), X_tensor), dim=1)

# Model with Batch Normalization
model_with_bn = nn.Sequential(
    nn.Flatten(),
    nn.Linear(2, 128),
    nn.BatchNorm1d(128),  # Batch Normalization layer
    nn.ReLU(),
    nn.Linear(128, 1)
)

# Model with Layer Normalization
model_with_ln = nn.Sequential(
    nn.Flatten(),
    nn.Linear(2, 128),
    nn.LayerNorm(128),  # Layer Normalization layer
    nn.ReLU(),
    nn.Linear(128, 1)
)

# Weight initialization function
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Apply weight initialization to both models
model_with_bn.apply(init_weights)
model_with_ln.apply(init_weights)

# Loss function
loss_fn = nn.MSELoss()

# Optimizer with the same learning rate as in TensorFlow
learning_rate = 0.001
optimizer_with_bn = optim.Adam(model_with_bn.parameters(), lr=learning_rate)
optimizer_with_ln = optim.Adam(model_with_ln.parameters(), lr=learning_rate)

# Training with Batch Normalization
history_with_bn = []
for epoch in range(100):
    y_pred = model_with_bn(X_b_tensor)
    loss = loss_fn(y_pred, y_tensor)
    optimizer_with_bn.zero_grad()
    loss.backward()
    optimizer_with_bn.step()
    history_with_bn.append(loss.item())

# Training with Layer Normalization
history_with_ln = []
for epoch in range(100):
    y_pred = model_with_ln(X_b_tensor)
    loss = loss_fn(y_pred, y_tensor)
    optimizer_with_ln.zero_grad()
    loss.backward()
    optimizer_with_ln.step()
    history_with_ln.append(loss.item())

# Plotting the loss curve
plt.title('Loss Convergence with Layer & Batch Normalization')
plt.plot(history_with_ln, label='Layer Normalization', color='orange')
plt.plot(history_with_bn, label='Batch Normalization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.legend()
plt.show()
