import os  # Importing the os module for operating system functionalities
import pandas as pd  # Importing pandas library with an alias pd for data manipulation and analysis
import torch  # Importing the PyTorch library for machine learning tasks
import torch.nn as nn  # Importing neural network modules from PyTorch
import torch.optim as optim  # Importing optimization algorithms from PyTorch
from torch.utils.data import DataLoader, Dataset  # Importing data loading utilities from PyTorch
from torchvision import transforms  # Importing transformation utilities from torchvision
from sklearn.model_selection import train_test_split  # Importing train_test_split function from scikit-learn
from sklearn.metrics import accuracy_score  # Importing accuracy_score function from scikit-learn
from PIL import Image  # Importing the Image module from the Python Imaging Library (PIL)
import matplotlib.pyplot as plt  # Importing the pyplot module from matplotlib for plotting

# Define a mapping from all labels to numeric labels
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,  # Mapping numeric labels 0-9
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,  # Mapping alphabetical labels A-Z, a-z
             'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
             'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
             'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
             'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55,
             'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

# Hyperparameters
input_size = 28 * 28  # Size of input features, assuming images are resized to 28x28 pixels
hidden_size = 128  # Number of neurons in the hidden layer of the neural network
num_classes = 62  # Total number of unique characters (digits, uppercase and lowercase letters), assuming 62 classes
learning_rate = 0.01  # Learning rate for the optimization algorithm
batch_size = 64  # Number of samples in each mini-batch during training
num_epochs = 50  # Number of complete passes through the entire dataset during training

# Data preprocessing
transform = transforms.Compose([  # Define a sequence of image transformations
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.Resize((28, 28)),  # Resize images to 28x28 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize image tensors
])


# Load the CSV file containing information about the dataset
data_info = pd.read_csv('english.csv')

# Define the path to the folder containing images
image_folder = 'img'

# Split the dataset into train, test, and validation sets using train_test_split function
train_data, temp_data = train_test_split(data_info, test_size=0.3, random_state=42)  # Splitting into train and temporary data
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # Further splitting temporary data into validation and test sets

# Save the split datasets to new CSV files
train_data.to_csv('train_data.csv', index=False)  # Saving train data to a CSV file without including index
valid_data.to_csv('valid_data.csv', index=False)  # Saving validation data to a CSV file without including index
test_data.to_csv('test_data.csv', index=False)  # Saving test data to a CSV file without including index

image_folder = '.'

# Load datasets using CustomDataset class
train_dataset = CustomDataset(csv_file='train_data.csv', root_dir=image_folder, transform=transform)  # Load training dataset
valid_dataset = CustomDataset(csv_file='valid_data.csv', root_dir=image_folder, transform=transform)  # Load validation dataset
test_dataset = CustomDataset(csv_file='test_data.csv', root_dir=image_folder, transform=transform)  # Load test dataset

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader for training dataset with shuffling
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)  # Create DataLoader for validation dataset without shuffling
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)  # Create DataLoader for test dataset without shuffling


# Initialize model, criterion, optimizer, and device
model = MLP(input_size, hidden_size, num_classes)  # Initialize the MLP model
criterion = nn.CrossEntropyLoss()  # Define the loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Define the optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose the device for training (GPU if available, else CPU)

# Initialize Trainer and start training
trainer = Trainer(model, criterion, optimizer, device)  # Initialize the Trainer object
train_losses, valid_losses, train_accuracies, valid_accuracies = trainer.train(train_loader, valid_loader, num_epochs)  # Start training the model


# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(valid_accuracies, label='Valid Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# Test the trained model
model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate accuracy on the test set
test_accuracy = accuracy_score(all_labels, all_preds) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Define a mapping from all labels to numeric labels
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
             'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
             'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
             'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
             'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55,
             'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_info.iloc[idx, 1]
        # Convert label to numeric label
        label = label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define Trainer class
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, valid_loader, num_epochs, writer):
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        min_valid_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted_train = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)
            writer.add_scalar('Loss/Train', train_loss, epoch+1)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch+1)

            # Validation
            self.model.eval()
            valid_loss = 0.0
            correct_valid = 0
            total_valid = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted_valid = torch.max(outputs, 1)
                    total_valid += labels.size(0)
                    correct_valid += (predicted_valid == labels).sum().item()

                valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(valid_loss)
                valid_accuracy = 100 * correct_valid / total_valid
                valid_accuracies.append(valid_accuracy)
                writer.add_scalar('Loss/Validation', valid_loss, epoch+1)
                writer.add_scalar('Accuracy/Validation', valid_accuracy, epoch+1)

                # Early stopping: Stop training if validation loss starts increasing
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                else:
                    print(f'Early stopping at epoch {epoch+1}, because validation loss started increasing.')
                    break

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        return train_losses, valid_losses, train_accuracies, valid_accuracies

# Hyperparameters
input_size = 28 * 28  # Assuming images are resized to 28x28
hidden_size = 128
num_classes = 62  # Assuming total number of unique characters is 62
learning_rate = 0.0001
batch_size = 64
num_epochs = 50 # Increase if needed

# Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])



# Load datasets
train_dataset = CustomDataset(csv_file='train_data.csv', root_dir=image_folder, transform=transform)
valid_dataset = CustomDataset(csv_file='valid_data.csv', root_dir=image_folder, transform=transform)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer, and device
model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize TensorBoard writer
writer = SummaryWriter()

# Initialize Trainer and start training
trainer = Trainer(model, criterion, optimizer, device)
train_losses, valid_losses, train_accuracies, valid_accuracies = trainer.train(train_loader, valid_loader, num_epochs, writer)

# Close TensorBoard writer
writer.close()

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(valid_accuracies, label='Valid Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# Test the trained model
test_dataset = CustomDataset(csv_file='test_data.csv', root_dir=image_folder, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate accuracy on the test set
test_accuracy = accuracy_score(all_labels, all_preds) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')


import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Define a mapping from all labels to numeric labels
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
             'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
             'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
             'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
             'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55,
             'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_info.iloc[idx, 1]
        # Convert label to numeric label
        label = label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Define Trainer class
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, valid_loader, num_epochs, writer):
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        min_valid_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted_train = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)
            writer.add_scalar('Loss/Train', train_loss, epoch+1)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch+1)

            # Validation
            self.model.eval()
            valid_loss = 0.0
            correct_valid = 0
            total_valid = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted_valid = torch.max(outputs, 1)
                    total_valid += labels.size(0)
                    correct_valid += (predicted_valid == labels).sum().item()

                valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(valid_loss)
                valid_accuracy = 100 * correct_valid / total_valid
                valid_accuracies.append(valid_accuracy)
                writer.add_scalar('Loss/Validation', valid_loss, epoch+1)
                writer.add_scalar('Accuracy/Validation', valid_accuracy, epoch+1)

                # Early stopping: Stop training if validation loss starts increasing
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                else:
                    print(f'Early stopping at epoch {epoch+1}, because validation loss started increasing.')
                    break

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        return train_losses, valid_losses, train_accuracies, valid_accuracies

# Hyperparameters
input_size = 28 * 28  # Assuming images are resized to 28x28
hidden_size1 = 256
hidden_size2 = 128
num_classes = 62  # Assuming total number of unique characters is 62
learning_rate = 0.001
batch_size = 64
num_epochs = 50  # Increase if needed

# Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])



# Load datasets
train_dataset = CustomDataset(csv_file='train_data.csv', root_dir=image_folder, transform=transform)
valid_dataset = CustomDataset(csv_file='valid_data.csv', root_dir=image_folder, transform=transform)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer, and device
model = MLP(input_size, hidden_size1, hidden_size2, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize TensorBoard writer
writer = SummaryWriter()

# Initialize Trainer and start training
trainer = Trainer(model, criterion, optimizer, device)
train_losses, valid_losses, train_accuracies, valid_accuracies = trainer.train(train_loader, valid_loader, num_epochs, writer)

# Close TensorBoard writer
writer.close()

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(valid_accuracies, label='Valid Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Test the trained model
test_dataset = CustomDataset(csv_file='test_data.csv', root_dir=image_folder, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate accuracy on the test set
test_accuracy = accuracy_score(all_labels, all_preds) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')


import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Define a mapping from all labels to numeric labels
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
             'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
             'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
             'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
             'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55,
             'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_info.iloc[idx, 1]
        # Convert label to numeric label
        label = label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Define Trainer class
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, valid_loader, num_epochs, writer):
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        min_valid_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted_train = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)
            writer.add_scalar('Loss/Train', train_loss, epoch+1)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch+1)

            # Validation
            self.model.eval()
            valid_loss = 0.0
            correct_valid = 0
            total_valid = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted_valid = torch.max(outputs, 1)
                    total_valid += labels.size(0)
                    correct_valid += (predicted_valid == labels).sum().item()

                valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(valid_loss)
                valid_accuracy = 100 * correct_valid / total_valid
                valid_accuracies.append(valid_accuracy)
                writer.add_scalar('Loss/Validation', valid_loss, epoch+1)
                writer.add_scalar('Accuracy/Validation', valid_accuracy, epoch+1)

                # Early stopping: Stop training if validation loss starts increasing
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                else:
                    print(f'Early stopping at epoch {epoch+1}, because validation loss started increasing.')
                    break

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        return train_losses, valid_losses, train_accuracies, valid_accuracies

# Hyperparameters
input_size = 28 * 28  # Assuming images are resized to 28x28
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
num_classes = 62  # Assuming total number of unique characters is 62
learning_rate = 0.001
batch_size = 64
num_epochs = 50  # Increase if needed

# Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])



# Load datasets
train_dataset = CustomDataset(csv_file='train_data.csv', root_dir=image_folder, transform=transform)
valid_dataset = CustomDataset(csv_file='valid_data.csv', root_dir=image_folder, transform=transform)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer, and device
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize TensorBoard writer
writer = SummaryWriter()

# Initialize Trainer and start training
trainer = Trainer(model, criterion, optimizer, device)
train_losses, valid_losses, train_accuracies, valid_accuracies = trainer.train(train_loader, valid_loader, num_epochs, writer)

# Close TensorBoard writer
writer.close()

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(valid_accuracies, label='Valid Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Test the trained model
test_dataset = CustomDataset(csv_file='test_data.csv', root_dir=image_folder, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate accuracy on the test set
test_accuracy = accuracy_score(all_labels, all_preds) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')


import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Define a mapping from all labels to numeric labels
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
             'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
             'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
             'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
             'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55,
             'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_info.iloc[idx, 1]
        # Convert label to numeric label
        label = label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Define Trainer class
class Trainer:
    def __init__(self, model, criterion, optimizer, device, patience):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.counter = 0
        self.best_valid_loss = float('inf')

    def train(self, train_loader, valid_loader, num_epochs, writer):
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted_train = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)
            writer.add_scalar('Loss/Train', train_loss, epoch+1)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch+1)

            # Validation
            self.model.eval()
            valid_loss = 0.0
            correct_valid = 0
            total_valid = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted_valid = torch.max(outputs, 1)
                    total_valid += labels.size(0)
                    correct_valid += (predicted_valid == labels).sum().item()

                valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(valid_loss)
                valid_accuracy = 100 * correct_valid / total_valid
                valid_accuracies.append(valid_accuracy)
                writer.add_scalar('Loss/Validation', valid_loss, epoch+1)
                writer.add_scalar('Accuracy/Validation', valid_accuracy, epoch+1)

                # Early stopping
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print(f'Early stopping at epoch {epoch+1}, because validation loss started increasing.')
                        break

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        return train_losses, valid_losses, train_accuracies, valid_accuracies

# Hyperparameters
input_size = 28 * 28  # Assuming images are resized to 28x28
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
num_classes = 62  # Assuming total number of unique characters is 62
learning_rate = 0.001
batch_size = 64
num_epochs = 50  # Increase if needed
patience = 10  # Number of epochs to wait before early stopping

# Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])



# Load datasets
train_dataset = CustomDataset(csv_file='train_data.csv', root_dir=image_folder, transform=transform)
valid_dataset = CustomDataset(csv_file='valid_data.csv', root_dir=image_folder, transform=transform)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer, and device
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize TensorBoard writer
writer = SummaryWriter()

# Initialize Trainer and start training
trainer = Trainer(model, criterion, optimizer, device, patience)
train_losses, valid_losses, train_accuracies, valid_accuracies = trainer.train(train_loader, valid_loader, num_epochs, writer)

# Close TensorBoard writer
writer.close()

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(valid_accuracies, label='Valid Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Test the trained model
test_dataset = CustomDataset(csv_file='test_data.csv', root_dir=image_folder, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate accuracy on the test set
test_accuracy = accuracy_score(all_labels, all_preds) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')


import torch  # Importing the PyTorch library for machine learning tasks
from torchvision import transforms  # Importing transformation utilities from torchvision
from PIL import Image, ImageDraw  # Importing the Image and ImageDraw modules from the Python Imaging Library (PIL)
import matplotlib.pyplot as plt  # Importing the pyplot module from matplotlib for plotting

# Load the input image
input_image_path = 'Img/img001-003.png'  # Path to the input image
input_image = Image.open(input_image_path)  # Opening the input image using PIL

# Define the transformation for preprocessing the input image
transform = transforms.Compose([  # Define a sequence of image transformations
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.Resize((28, 28)),  # Resize images to 28x28 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize image tensors
])

# Preprocess the input image
input_image_tensor = transform(input_image).unsqueeze(0)  # Add a batch dimension to the preprocessed image

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose the device for processing (GPU if available, else CPU)

# Pass the preprocessed image through the model to get predictions
with torch.no_grad():  # No gradient computation needed
    input_image_tensor = input_image_tensor.to(device)  # Move data to device
    outputs = model(input_image_tensor)  # Forward pass through the model
    _, predicted_class = torch.max(outputs, 1)  # Get the predicted class index
    predicted_class = predicted_class.item()  # Convert predicted class to Python scalar

# Get the predicted label and probability
predicted_label = list(label_map.keys())[list(label_map.values()).index(predicted_class)]  # Map predicted class index to label
predicted_probability = torch.softmax(outputs, dim=1)[0, predicted_class].item()  # Get the probability of the predicted class

# Determine the bounding box coordinates (you need to define these based on your requirements)
# For example, let's assume the predicted character is centered in the image
image_width, image_height = input_image.size  # Get the dimensions of the input image
box_width, box_height = 500, 600  # Define the size of the bounding box
left = (image_width - box_width) // 2  # Calculate left coordinate of the bounding box
top = (image_height - box_height) // 2  # Calculate top coordinate of the bounding box
right = left + box_width  # Calculate right coordinate of the bounding box
bottom = top + box_height  # Calculate bottom coordinate of the bounding box

# Visualize the prediction on the input image with a bounding box and larger text
plt.figure(figsize=(8, 8))  # Set the size of the plot
plt.imshow(input_image)  # Display the input image
plt.axis('off')  # Turn off axis

# Draw bounding box
plt.plot([left, left], [top, bottom], color='red', linewidth=2)  # Draw left edge of the bounding box
plt.plot([right, right], [top, bottom], color='red', linewidth=2)  # Draw right edge of the bounding box
plt.plot([left, right], [top, top], color='red', linewidth=2)  # Draw top edge of the bounding box
plt.plot([left, right], [bottom, bottom], color='red', linewidth=2)  # Draw bottom edge of the bounding box

# Add text
text = f'{predicted_label} ({predicted_probability:.2f})'  # Text to display (label and probability)
font_size = 16  # Adjust the font size as needed
plt.text((left + right) / 2, top - 20, text, fontsize=font_size * 2, color='red', ha='center')  # Display the text at the top of the bounding box

# Show the plot
plt.show()  # Display the plot


import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Define a mapping from all labels to numeric labels
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
             'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
             'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
             'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
             'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55,
             'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_info.iloc[idx, 1]
        # Convert label to numeric label
        label = label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First fully connected layer
        self.relu1 = nn.ReLU()  # ReLU activation function after first layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second fully connected layer
        self.relu2 = nn.ReLU()  # ReLU activation function after second layer
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  # Third fully connected layer
        self.relu3 = nn.ReLU()  # ReLU activation function after third layer
        self.fc4 = nn.Linear(hidden_size3, num_classes)  # Fourth fully connected layer (output layer)
        self.dropout = nn.Dropout(dropout)  # Dropout layer to prevent overfitting
        self.batch_norm1 = nn.BatchNorm1d(hidden_size1)  # Batch normalization layer after first layer
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)  # Batch normalization layer after second layer
        self.batch_norm3 = nn.BatchNorm1d(hidden_size3)  # Batch normalization layer after third layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)  # Pass through first fully connected layer
        x = self.batch_norm1(x)  # Apply batch normalization
        x = self.relu1(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Pass through second fully connected layer
        x = self.batch_norm2(x)  # Apply batch normalization
        x = self.relu2(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Pass through third fully connected layer
        x = self.batch_norm3(x)  # Apply batch normalization
        x = self.relu3(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc4(x)  # Pass through fourth fully connected layer (output layer)
        return x  # Return the output


# Define Trainer class
class Trainer:
    def __init__(self, model, criterion, optimizer, device, patience):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.counter = 0
        self.best_valid_loss = float('inf')

    def train(self, train_loader, valid_loader, num_epochs, writer):
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted_train = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)
            writer.add_scalar('Loss/Train', train_loss, epoch+1)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch+1)

            # Validation
            self.model.eval()
            valid_loss = 0.0
            correct_valid = 0
            total_valid = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    _, predicted_valid = torch.max(outputs, 1)
                    total_valid += labels.size(0)
                    correct_valid += (predicted_valid == labels).sum().item()

                valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(valid_loss)
                valid_accuracy = 100 * correct_valid / total_valid
                valid_accuracies.append(valid_accuracy)
                writer.add_scalar('Loss/Validation', valid_loss, epoch+1)
                writer.add_scalar('Accuracy/Validation', valid_accuracy, epoch+1)

                # Early stopping
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print(f'Early stopping at epoch {epoch+1}, because validation loss started increasing.')
                        break

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        return train_losses, valid_losses, train_accuracies, valid_accuracies

# Hyperparameters
input_size = 28 * 28  # Assuming images are resized to 28x28
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
num_classes = 62  # Assuming total number of unique characters is 62
learning_rate = 0.01
batch_size = 64
num_epochs = 50  # Increase if needed
patience = 10  # Number of epochs to wait before early stopping

# Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])



# Load datasets
train_dataset = CustomDataset(csv_file='train_data.csv', root_dir=image_folder, transform=transform)
valid_dataset = CustomDataset(csv_file='valid_data.csv', root_dir=image_folder, transform=transform)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer, and device
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize TensorBoard writer
writer = SummaryWriter()

# Initialize Trainer and start training
trainer = Trainer(model, criterion, optimizer, device, patience)
train_losses, valid_losses, train_accuracies, valid_accuracies = trainer.train(train_loader, valid_loader, num_epochs, writer)

# Close TensorBoard writer
writer.close()

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(valid_accuracies, label='Valid Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Test the trained model
test_dataset = CustomDataset(csv_file='test_data.csv', root_dir=image_folder, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate accuracy on the test set
test_accuracy = accuracy_score(all_labels, all_preds) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')


import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load the input image
input_image_path = 'Img/img001-003.png'
input_image = Image.open(input_image_path)

# Define the transformation for preprocessing the input image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# Preprocess the input image
input_image_tensor = transform(input_image).unsqueeze(0)  # Add a batch dimension

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pass the preprocessed image through the model to get predictions
with torch.no_grad():
    input_image_tensor = input_image_tensor.to(device)
    outputs = model(input_image_tensor)
    _, predicted_class = torch.max(outputs, 1)
    predicted_class = predicted_class.item()

# Get the predicted label and probability
predicted_label = list(label_map.keys())[list(label_map.values()).index(predicted_class)]
predicted_probability = torch.softmax(outputs, dim=1)[0, predicted_class].item()

# Determine the bounding box coordinates (you need to define these based on your requirements)
# For example, let's assume the predicted character is centered in the image
image_width, image_height = input_image.size
box_width, box_height = 500, 600  # Increase the size of the bounding box
left = (image_width - box_width) // 2
top = (image_height - box_height) // 2
right = left + box_width
bottom = top + box_height

# Visualize the prediction on the input image with a bounding box and larger text
plt.figure(figsize=(8, 8))
plt.imshow(input_image)
plt.axis('off')

# Draw bounding box
plt.plot([left, left], [top, bottom], color='red', linewidth=2)
plt.plot([right, right], [top, bottom], color='red', linewidth=2)
plt.plot([left, right], [top, top], color='red', linewidth=2)
plt.plot([left, right], [bottom, bottom], color='red', linewidth=2)

# Add text
text = f'{predicted_label} ({predicted_probability:.2f})'
font_size = 16  # Adjust the font size as needed
plt.text((left + right) / 2, top - 20, text, fontsize=font_size * 2, color='red', ha='center')

# Show the plot
plt.show()


import cv2  # Importing OpenCV library for image processing
import torch  # Importing the PyTorch library for machine learning tasks
from torchvision import transforms  # Importing transformation utilities from torchvision
import matplotlib.pyplot as plt  # Importing the pyplot module from matplotlib for plotting
from PIL import Image  # Importing the Image module from the Python Imaging Library (PIL)

# Load the input image
input_image_path = 'new2.png'  # Path to the input image
input_image = cv2.imread(input_image_path)  # Reading the input image using OpenCV

# Convert image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # Converting the input image to grayscale

# Threshold the grayscale image to obtain binary image (1 for black, 0 for white)
_, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)  # Thresholding the grayscale image

# Invert the binary image to have 1 for white and 0 for black
binary_image = cv2.bitwise_not(binary_image)  # Inverting the binary image

# Apply contour detection
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Detecting contours in the binary image

# Define the transformation for preprocessing the input image
transform = transforms.Compose([  # Define a sequence of image transformations
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.Resize((28, 28)),  # Resize images to 28x28 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize image tensors
])

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose the device for processing (GPU if available, else CPU)

# Define a mapping from all labels to numeric labels
label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
             'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
             'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
             'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
             'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55,
             'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

# Define a mapping from numeric labels to characters
label_map_inverse = {value: key for key, value in label_map.items()}

# Iterate over contours
for contour in contours:
    # Get bounding box coordinates for the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract region of interest (ROI) from the original image
    roi = input_image[y:y+h, x:x+w]
    
    # Preprocess the ROI
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)  # Preprocess and move to device
    
    # Pass the preprocessed ROI through the model to get predictions
    with torch.no_grad():  # No gradient computation needed
        outputs = model(roi_tensor)  # Forward pass through the model
        _, predicted_class = torch.max(outputs, 1)  # Get the predicted class index
        predicted_class = predicted_class.item()  # Convert predicted class to Python scalar
    
    # Get the predicted label and probability
    predicted_label = label_map_inverse[predicted_class]  # Map predicted class index to label
    predicted_probability = torch.softmax(outputs, dim=1)[0, predicted_class].item()  # Get the probability of the predicted class
    
    # Draw bounding box around the contour
    cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
    
    # Display label and probability
    label_text = f'{predicted_label} ({predicted_probability:.2f})'  # Text to display (label and probability)
    cv2.putText(input_image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add text to the image

# Display the result
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))  # Display the result image
plt.axis('off')  # Turn off axis
plt.show()  # Show the plot


