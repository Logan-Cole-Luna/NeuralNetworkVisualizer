import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class ConvPerceptron(nn.Module):
    def __init__(self):
        super(ConvPerceptron, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Load and preprocess the image
image_path = 'PlaneImage.jpg'
image = Image.open(image_path).convert('RGB')

# Define a transformation to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Apply the transformation to the image
input_image = preprocess(image).unsqueeze(0)

# Load the model
model = ConvPerceptron()

# Pass the image through the model
output = model(input_image)

# Assuming you have the ground truth label, let's say it's class index 3
ground_truth = 3

# Calculate the predicted class
predicted_class = torch.argmax(output, dim=1).item()

# Check if the prediction matches the ground truth
correct = predicted_class == ground_truth

# Print the predicted output, predicted class, and accuracy
print("Predicted Output:", output)
print("Predicted Class:", predicted_class)
print("Accuracy:", "Correct" if correct else "Incorrect")
