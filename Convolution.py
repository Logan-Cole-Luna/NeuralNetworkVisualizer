import numpy as np
from PIL import Image

# Load the image using PIL (Python Imaging Library)
image_path = 'PlaneImage.jpg'
image = Image.open(image_path).convert('L')  # Convert to grayscale

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

kernel2 = np.array([[9, 9, 9],
                   [9, 9, 9],
                   [9, 9, 9]])

# Convert the image and kernel to NumPy arrays
image_array = np.array(image)
kernel_array = np.array(kernel)

# Get the dimensions of the image and kernel
image_height, image_width = image_array.shape
kernel_height, kernel_width = kernel_array.shape

# Initialize an empty output image
output_image = np.zeros_like(image_array)

# Perform convolution
for y in range(image_height - kernel_height + 1):
    for x in range(image_width - kernel_width + 1):
        image_patch = image_array[y:y+kernel_height, x:x+kernel_width]
        output_image[y, x] = np.sum(image_patch) * 9

# Clip the pixel values to ensure they stay in the valid range [0, 255]
output_image = np.clip(output_image, 0, 255)

# Convert the output image back to uint8 data type
output_image = output_image.astype(np.uint8)

# Save and display the output image using PIL
output_image_pil = Image.fromarray(output_image)
output_image_pil.show()
output_image_pil.save('output_image.jpg')

