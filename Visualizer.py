import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

user_choice = 0
while user_choice != 1 and user_choice != 2 and user_choice != 3:
    user_choice = int(
        input("Would you like to visualize the network on XOR Dataset(1),  Heart Disease Dataset(2), or both(3)? "))
    print(user_choice)
    if user_choice != 1 and user_choice != 2 and user_choice != 3:
        print("Incorrect input, please try again")

# Activation functions
activation_functions = {
    1: F.sigmoid,
    2: F.relu,
    3: F.tanh,
    4: F.leaky_relu
}

# Loss functions
loss_functions = {
    1: nn.MSELoss(),
    2: nn.CrossEntropyLoss()
}

# Mapping of activation_id and loss_id to titles
title_mapping = {
    (1, 1): 'Mean Squared Error Loss with Sigmoid',
    (1, 2): 'Cross Entropy Loss with Sigmoid',
    (2, 1): 'Mean Squared Error Loss with ReLU',
    (2, 2): 'Cross Entropy Loss with ReLU',
    (3, 1): 'Mean Squared Error Loss with Tanh',
    (3, 2): 'Cross Entropy Loss with Tanh',
    (4, 1): 'Mean Squared Error Loss with Leaky ReLU',
    (4, 2): 'Cross Entropy Loss with Leaky ReLU'
}

# Number of neurons and layers in the model
num_layers = 2
num_neurons = 3

successfulNetworksXOR = []
successfulNetworksHeart = []


def xor_network():
    print("XOR Gate Identifier")
    # Input data
    X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = torch.Tensor([0, 1, 1, 0]).view(-1, 1)

    # Iterate over activation functions and loss functions
    for activation_id, activation_func in activation_functions.items():
        for loss_id, loss_func in loss_functions.items():
            # Define the XOR model
            class XOR(nn.Module):
                def __init__(self, input_dim=2, output_dim=1):
                    super(XOR, self).__init__()
                    self.layers = nn.ModuleList([nn.Linear(input_dim, num_neurons)])
                    self.layers.extend([nn.Linear(num_neurons, num_neurons) for _ in range(num_layers - 1)])
                    self.lin_out = nn.Linear(num_neurons, output_dim)

                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                        x = activation_func(x)
                    x = self.lin_out(x)
                    return x

            model = XOR()

            # Function to initialize model weights
            def weights_init(model):
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data.normal_(0, 1)

            weights_init(model)

            optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

            epochs = 2000
            loss_history = []
            accuracy_history = []

            for i in range(epochs):
                for j in range(len(X)):
                    data_point = np.random.randint(len(X))
                    x_var = Variable(X[data_point], requires_grad=False)
                    y_var = Variable(Y[data_point], requires_grad=False)

                    optimizer.zero_grad()
                    y_hat = model(x_var)
                    loss = loss_func(y_hat, y_var)
                    loss.backward()
                    optimizer.step()

                loss_history.append(loss.item())

                predictions = (model(X) > 0.5).float()
                accuracy = (predictions == Y).float().mean()

                accuracy_history.append(accuracy.item())

                print_results(i, loss.item(), accuracy.item(), activation_id, loss_id)

            if accuracy_history[-1] == 1.0:
                successfulNetworksXOR.append(title_mapping.get((activation_id, loss_id), "Unknown"))

            model_params = list(model.parameters())
            model_weights = model_params[0].data.numpy()
            model_bias = model_params[1].data.numpy()

            # Plot results
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.scatter(X.numpy()[[0, -1], 0], X.numpy()[[0, -1], 1], s=50)
            plt.scatter(X.numpy()[[1, 2], 0], X.numpy()[[1, 2], 1], c='red', s=50)

            x_1 = np.arange(-0.1, 1.1, 0.1)
            y_1 = ((x_1 * model_weights[0, 0]) + model_bias[0]) / (-model_weights[0, 1])
            plt.plot(x_1, y_1)

            x_2 = np.arange(-0.1, 1.1, 0.1)
            y_2 = ((x_2 * model_weights[1, 0]) + model_bias[1]) / (-model_weights[1, 1])
            plt.plot(x_2, y_2)
            plt.legend([f"neuron_{i + 1}" for i in range(num_neurons)], loc=8)

            title = "XOR Gate with ", title_mapping.get((activation_id, loss_id), "Unknown")

            plt.title(title)

            plt.subplot(1, 2, 2)
            plt.plot(range(epochs), loss_history)
            plt.plot(range(epochs), accuracy_history)
            plt.legend(["Loss", "Accuracy"], loc="upper right")
            plt.title(title)

            plt.tight_layout()
            plt.show()

    print("Successful Networks For XOR: ", successfulNetworksXOR)
    print("\n\n")


def heart_network():
    print("Heart Disease Identifier")

    # Heart disease identifier training
    data = pd.read_csv('heart.csv')
    X = data.drop('target', axis=1)
    Y = data['target']
    X_encoded = pd.get_dummies(X)
    X_encoded = X_encoded.astype(float)
    X_encoded = (X_encoded - X_encoded.mean()) / X_encoded.std()
    X_encoded.fillna(0, inplace=True)
    X = X_encoded.values
    Y = Y.values
    input_dim = X.shape[1]

    # Iterate over activation functions and loss functions
    for activation_id, activation_func in activation_functions.items():
        for loss_id, loss_func in loss_functions.items():
            class Heart(nn.Module):
                def __init__(self, input_dim, output_dim, num_neurons, num_layers):
                    super(Heart, self).__init__()
                    self.layers = nn.ModuleList([nn.Linear(input_dim, num_neurons)])
                    self.layers.extend([nn.Linear(num_neurons, num_neurons) for _ in range(num_layers - 1)])
                    self.lin_out = nn.Linear(num_neurons, output_dim)

                def forward(self, x):
                    for layer in self.layers:
                        x = activation_func(layer(x))
                    x = self.lin_out(x)
                    return x

            heart_model = Heart(input_dim, 2, 8, 2)

            # Function to initialize model weights
            def weights_init(model):
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data.normal_(0, 1)

            weights_init(heart_model)

            heart_optimizer = optim.Adam(heart_model.parameters(), lr=0.001)
            heart_loss_func = loss_functions[2]
            epoch = 2001
            heart_losses = []
            heart_accuracy = []

            X_tensor = torch.tensor(X, dtype=torch.float32)
            Y_tensor = torch.tensor(Y, dtype=torch.long)

            for epoch in range(epoch):
                heart_optimizer.zero_grad()
                heart_outputs = heart_model(X_tensor)
                heart_loss = heart_loss_func(heart_outputs, Y_tensor)
                heart_loss.backward()
                heart_optimizer.step()
                heart_losses.append(heart_loss.item())
                # Heart disease identifier predictions
                heart_predictions = torch.argmax(heart_model(X_tensor), dim=1)
                heart_correct = (heart_predictions == Y_tensor).sum().item()
                heart_accuracy = heart_correct / len(Y_tensor) * 100

                print_results(epoch, heart_loss.item(), heart_accuracy, activation_id, loss_id)

            if heart_accuracy >= 9.0:
                successfulNetworksHeart.append(title_mapping.get((activation_id, loss_id), "Unknown"))

            print("\nHeart Disease Identifier Results:")
            print(f"Predicted: {heart_predictions}")
            print(f"Target:    {Y_tensor}")
            print(f"Accuracy:  {heart_accuracy}%")

            # Plot the loss & accuracy curves
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(heart_losses)
            plt.plot(heart_accuracy)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(["Loss", "Accuracy"], loc="upper right")
            title = "Heart Disease Identifier Loss Curve with " + title_mapping.get((activation_id, loss_id), "Unknown")
            plt.title(title)

            # Plot the Heart dataset
            plt.subplot(1, 2, 2)
            plt.tight_layout()
            plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Heart Disease Identifier')
            plt.show()

    print("Successful Networks For Heart Disease Identifier: ", successfulNetworksHeart)
    print("\n\n")


def print_results(epoch, loss, accuracy, activation_id, loss_id):
    if epoch % 500 == 0:
        if epoch == 0:
            print(title_mapping.get((activation_id, loss_id)))
            print("--------------------------------------")
        print("Epoch: {0}, Loss: {1}, Accuracy: {2}".format(epoch, loss, accuracy))
        if epoch == 2000:
            print()


if user_choice == 1:
    xor_network()
# uh
elif user_choice == 2:
    heart_network()

elif user_choice == 3:
    xor_network()
    heart_network()
else:
    print("Error")
print("End")
