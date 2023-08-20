import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generate some example data for Categorical Cross-Entropy Loss
num_classes = 5
logits = torch.randn(num_classes)  # Logits for a single sample
target_label = torch.randint(0, num_classes, ())  # Target label for a single sample

# Define the Categorical Cross-Entropy loss function
cce_loss_function = nn.CrossEntropyLoss()

# Calculate the CCE loss
cce_loss = cce_loss_function(logits.unsqueeze(0), target_label.unsqueeze(0))

# Generate some example data for Mean Squared Error (MSE) Loss
predicted_value = torch.randn(1)  # Predicted value for a single sample
target_value = torch.randn(1)  # Target value for a single sample

# Define the Mean Squared Error (MSE) loss function
mse_loss_function = nn.MSELoss()

# Calculate the MSE loss
mse_loss = mse_loss_function(predicted_value, target_value)

# Plot the loss functions
plt.figure(figsize=(10, 5))

# Plot the Categorical Cross-Entropy Loss
plt.subplot(1, 2, 1)
plt.plot(logits.detach().numpy(), 'bo', label='Logits')
plt.axvline(x=target_label.item(), color='r', linestyle='--', label='Target Label')
plt.xlabel('Class Index')
plt.ylabel('Logits')
plt.title('Categorical Cross-Entropy Loss')
plt.legend()

# Plot the Mean Squared Error (MSE) Loss
plt.subplot(1, 2, 2)
plt.plot(predicted_value.detach().numpy(), 'bo', label='Predicted Value')
plt.axhline(y=target_value.item(), color='r', linestyle='--', label='Target Value')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Mean Squared Error (MSE) Loss')
plt.legend()

plt.tight_layout()
plt.show()
