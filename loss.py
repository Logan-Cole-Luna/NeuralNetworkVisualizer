import numpy as np
import matplotlib.pyplot as plt

# Generating random loss values for visualization
num_epochs = 50
loss_values = np.random.rand(num_epochs)

# Plotting the loss values over epochs
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()
