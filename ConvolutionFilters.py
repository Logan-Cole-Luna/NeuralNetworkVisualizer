import numpy as np
import matplotlib.pyplot as plt

# Generating random convolution filters for visualization
num_filters = 16
filter_size = 3
filters = np.random.rand(filter_size, filter_size, num_filters)

plt.figure(figsize=(8, 8))
for i in range(num_filters):
    plt.subplot(4, 4, i+1)
    plt.imshow(filters[:, :, i], cmap='gray')
    plt.axis('off')
plt.show()
