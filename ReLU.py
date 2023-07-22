import numpy as np
import matplotlib.pyplot as plt

# Define the range for x (-5 to 5)
x = np.linspace(-5, 5, 100)

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU function
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# Plotting the ReLU and Leaky ReLU functions
plt.plot(x, relu(x), label='ReLU', color='orange')
plt.plot(x, leaky_relu(x), label='Leaky ReLU', linestyle='dashed', color='blue')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU and Leaky ReLU Activation Functions')
plt.legend()
plt.grid(True)
plt.show()




