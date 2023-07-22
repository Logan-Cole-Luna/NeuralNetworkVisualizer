import numpy as np
import matplotlib.pyplot as plt

# Sample data generation
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Adding a bias term to the input feature X
X_b = np.c_[np.ones((100, 1)), X]



# Initial parameters for the linear regression model
theta = np.random.randn(2, 1)

# Learning rate for both Gradient Descent and Stochastic Gradient Descent
learning_rate = 0.1

# Number of iterations
n_iterations = 100

# Lists to store the loss values during optimization
loss_gd = []
loss_sgd = []

# Gradient Descent
for iteration in range(n_iterations):
    gradients = -2 / len(X) * X_b.T.dot(y - X_b.dot(theta))
    theta -= learning_rate * gradients
    loss = np.mean((X_b.dot(theta) - y) ** 2)
    loss_gd.append(loss)

# Stochastic Gradient Descent
theta = np.random.randn(2, 1)
for iteration in range(n_iterations * len(X)):
    random_index = np.random.randint(len(X))
    xi = X_b[random_index:random_index + 1]
    yi = y[random_index:random_index + 1]
    gradients = -2 * xi.T.dot(yi - xi.dot(theta))
    theta -= learning_rate * gradients
    loss = np.mean((X_b.dot(theta) - y) ** 2)
    loss_sgd.append(loss)

# Creating subplots for both Gradient Descent and Stochastic Gradient Descent
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the loss values for Gradient Descent
axs[0].plot(range(n_iterations), loss_gd, label='Gradient Descent')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss Convergence: Gradient Descent')
axs[0].legend()
axs[0].grid(True)

# Plotting the loss values for Stochastic Gradient Descent
axs[1].plot(range(n_iterations * len(X)), loss_sgd, label='Stochastic Gradient Descent', linestyle='dashed', color='orange')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Loss')
axs[1].set_title('Loss Convergence: Stochastic Gradient Descent')
axs[1].legend()
axs[1].grid(True)

# Remove overlapping axes
plt.tight_layout()

# Show the plots
plt.show()

# Plotting the loss curve
plt.title('Loss Convergence with Layer & Batch Normalization')
plt.plot(range(n_iterations * len(X)), loss_sgd, label='Stochastic Gradient Descent', linestyle='dashed', color='orange')
plt.plot(range(n_iterations), loss_gd, label='Gradient Descent')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.legend()
plt.show()
