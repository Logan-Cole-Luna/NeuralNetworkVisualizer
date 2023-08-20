from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import gzip
from torchvision.io import read_image
from torch.utils.data import Dataset

from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax
from torch import flatten


# Define LeNet architecture as a PyTorch Module
class LeNet(Module):
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()
        # Define layers of the LeNet architecture
        # Conv Layer
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        # Pooling Layer 1
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        # Pooling Layer 2
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()
        # Softmax Layer
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output


class CustomDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            image_file = os.path.join(root, "raw", "train-images-idx3-ubyte.gz")
            label_file = os.path.join(root, "raw", "train-labels-idx1-ubyte.gz")
        else:
            image_file = os.path.join(root, "raw", "t10k-images-idx3-ubyte.gz")
            label_file = os.path.join(root, "raw", "t10k-labels-idx1-ubyte.gz")

        with gzip.open(image_file, "rb") as f:
            self.images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28, 1)

        with gzip.open(label_file, "rb") as f:
            self.labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


def plot(H):
    # Plot and save the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    # Display lists previously appended
    plt.subplot(1, 2, 1)
    plt.plot(H["train_loss"], label="train_loss", color="blue", linewidth=2)
    plt.plot(H["val_loss"], label="val_loss", color="green", linestyle="-.", linewidth=2)
    plt.legend(loc="lower left")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")

    plt.title("Training Loss on Dataset")
    plt.subplot(1, 2, 2)
    plt.plot(H["train_acc"], label="train_acc", color="red", linestyle="--", linewidth=2)
    plt.plot(H["val_acc"], label="val_acc", color="purple", linestyle=":", linewidth=2)
    plt.legend(loc="lower left")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy on Dataset")

    plt.tight_layout()
    plt.show()


def train():
    print("[INFO] training the network...")
    startTime = time.time()

    for e in range(0, EPOCHS):
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0

        print(f"Epoch {e + 1}/{EPOCHS}")
        # input data x and corresponding target labels y.
        for i, (x, y) in enumerate(trainDataLoader):
            (x, y) = (x.to(device), y.to(device))
            # passes the input data x through the trained model to generate predictions pred
            pred = model(x)
            # Print training progress for each batch
            current_percent = (i + 1) / len(trainDataLoader) * 100
            # loss between the predicted values pred and the actual target labels y using the chosen loss function lossFn
            loss = lossFn(pred, y)
            # resets the gradients of the model's parameters.
            # Gradients need to be cleared before computing new gradients in the backpropagation step.
            opt.zero_grad()
            # performs backpropagation to compute the gradients of the loss with respect to the model's parameters.
            loss.backward()
            # updates the model's parameters based on the computed gradients and the chosen optimizer
            # (in this case, Adam optimizer) using the learning rate
            opt.step()
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            # Print training progress for each batch
            current_percent = (i + 1) / len(trainDataLoader) * 100
            # combination of \r and end=''achieves the effect of updating the printed content
            print(f"\r  Epoch Progress: {current_percent:.2f}%", end='')
            if i == len(trainDataLoader) - 1:
                print()  # Print a newline at the end of the epoch

        # Print training progress for each batch
        print(f"  Batch: {trainCorrect}/{len(trainDataLoader.dataset)}")

        with torch.no_grad():
            model.eval()

            for (x, y) in valDataLoader:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                totalValLoss += lossFn(pred, y)
                valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))


# Set hyperparameters and device
TRAIN_SPLIT = 0.75
BATCH_SIZE = 64
EPOCHS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset instances
train_dataset = CustomDataset(root="data/KMNIST", train=True, transform=ToTensor())
val_dataset = CustomDataset(root="data/KMNIST", train=False, transform=ToTensor())
test_dataset = CustomDataset(root="data/KMNIST", train=False, transform=ToTensor())

# Create data loaders
trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Set hyperparameters and device
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 3
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate the number of classes
num_classes = len(set(train_dataset.labels))

trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# Initialize the LeNet model, optimizer, loss function, and history
print("[INFO] initializing the LeNet model...")
model = LeNet(numChannels=1, classes=num_classes).to(device)
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()
H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# Start training
train()

# Evaluate the trained model on the test set
print("[INFO] evaluating network...")
model.eval()
preds = []

with torch.no_grad():
    for (x, y) in testDataLoader:
        x = x.to(device)
        pred = model(x)
        preds.extend(pred.argmax(1).cpu().numpy())

classes = [str(i) for i in range(num_classes)]
classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

# Print classification report for the test set
print(classification_report(test_dataset.labels, np.array(preds), target_names=classes))

# Plot data
plot(H)

# plt.savefig(args.plot)  # Save the plot to the specified path

# Save the trained model
# torch.save(model, args.model)  # Save the model to the specified path
