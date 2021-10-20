import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
import datetime

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor

# Set path to store datasets
# The dataset will be retrieved and used if present
# Dataset will be downloaded and stored if not present
dataset_path = '~/datasets/'

# Directory path to store model
model_dir = 'models'

class Net(torch.nn.Module):
    """
    A custom Neural Network class.

    Subclass of torch.nn.Module and implements the forward pass.
    """

    def __init__(self) -> None:
        """
        Constructor of the custom Net class.

        Defines all the layers of the network.
        """

        super(Net, self).__init__()
        # A simple network with 4 convolution layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding='same')
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding='same')
        self.fc1 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Computes the forward pass through the network.

        Args:
            x (Tensor): Input tensor to the network

        Returns:
            Tensor: Computed output of the network
        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        # Flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.25, training=self.training)
        # Return value of output layer without activation
        return self.fc1(x)

def train(epoch: int) -> None:
    """
    Training loop

    Trains the network over the training data. One call corresponds to one epoch.

    Args:
        epoch (int): Epoch number of the current call. Used for progress display
    """

    # Set training = True
    network.train()

    # Variables to store training metrics
    train_loss = 0
    correct = 0
    num_data = 0
    start_time = time.time()

    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        # Send images and labels to device
        data, target = data.to(device), target.to(device)

        # Compute network output and update network weights
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Store loss and accuracy
        train_loss += loss.item() * len(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        num_data += len(data)

        time_per_batch = (time.time() - start_time) / (batch_idx + 1)
        eta = (len(train_loader) - batch_idx - 1) * time_per_batch

        # Print loss and accuracy after every batch
        print_string = ('Train Epoch: {} [{}/{} ({:.0f}%)] - ETA: {:.0f}s - Loss: {:.6f} - Accuracy: {:.2f}%'.format(
            epoch, num_data, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), eta,
            train_loss / num_data, 100. * correct / num_data))
        sys.stdout.write('\r\033[K' + print_string)
        sys.stdout.flush()

    total_time = time.time() - start_time
    # Print final loss and accuracy
    print_string = ('Train Epoch: {} [{}/{} ({:.0f}%)] - Time: {} - Loss: {:.6f} - Accuracy: {:.2f}%'.format(
                epoch, num_data, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                str(datetime.timedelta(seconds=total_time)),
                train_loss / num_data, 100. * correct / num_data))
    sys.stdout.write('\r\033[K' + print_string + '\n')
    sys.stdout.flush()

def val() -> None:
    """
    Validation loop

    Computes and prints performance of the model on the validation data.
    """

    # Set training = False
    network.eval()

    # Variables for storing validation metrics
    val_loss = 0
    correct = 0

    # Gradient computation not required during validation
    with torch.no_grad():
        # Validation loop
        for data, target in val_loader:
            # Send images and labels to device
            data, target = data.to(device), target.to(device)

            # Compute output and loss
            output = network(data)
            val_loss += F.cross_entropy(output, target, size_average=False).item()

            # Check predictions and compute accuracy
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    # Compute average validation loss
    val_loss /= len(val_loader.dataset)

    # Print validation metrics
    print('Validation set: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def test() -> None:
    """
    Test loop

    Computes and prints performance of the model on the test data.
    """

    # Set training = False
    network.eval()

    # Variables for storing test metrics
    test_loss = 0
    correct = 0

    # Gradient computation not required during testing
    with torch.no_grad():
        # Test loop
        for data, target in test_loader:
            # Send images and labels to device
            data, target = data.to(device), target.to(device)

            # Compute output and loss
            output = network(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()

            # Check predictions and compute accuracy
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    # Compute average test loss
    test_loss /= len(test_loader.dataset)

    # Print test metrics
    print('\nTest set: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    # Hyperparameters
    n_epochs = 20
    batch_size = 64
    learning_rate = 0.001
    train_val_split = 0.8

    # Set seed for replicability
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ', device, '\n', sep='')

    # Load MNIST training dataset (download if not present)
    # The pixel values images are in the range [0, 1]
    full_dataset = torchvision.datasets.MNIST(dataset_path, train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]))

    # Split data into training and validation
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create training and validation dataloaders from the datasets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Load MNIST test data (download if not present)
    # The pixel values images are in the range [0, 1]
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(dataset_path, train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True)

    # Create the network object and send it to device
    network = Net()
    network.to(device)

    # Create Adam optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # Initial validation to check random output of the network
    val()

    # Training
    for epoch in range(1, n_epochs + 1):
        # Train over entire data
        train(epoch)

        # Print validation metrics after every training epoch
        val()

    # Print test metrics of the final model
    test()

    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save final model parameters and optimizer state
    torch.save(network.state_dict(), os.path.join(model_dir, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer.pth'))
