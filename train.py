import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import CNN

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
batch_size = 64
learning_rate = 0.01
num_epochs = 3

# load dataset
train_dataset = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the network
model = CNN(in_channels=1, num_classes=10).to(device=device)


# train the network
def train():
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward propagation
            outputs = model(data)
            loss = criterion(outputs, targets)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # updates the weights
            optimizer.step()

            if batch_idx % 200 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} Loss : {loss:.4f}")


# check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy of training set")
    else:
        print("Checking accuracy of test set")

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            output = model(x)
            _, predictions = output.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct) * 100 / float(num_samples):.2f}")

    model.train()


if __name__ == "__main__":
    train()
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
