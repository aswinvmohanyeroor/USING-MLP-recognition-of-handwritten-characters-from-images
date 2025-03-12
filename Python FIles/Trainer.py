# Define Trainer class
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)  # Moving the model to the specified device (CPU or GPU)
        self.criterion = criterion  # Loss function
        self.optimizer = optimizer  # Optimization algorithm
        self.device = device  # Device for training (CPU or GPU)

    def train(self, train_loader, valid_loader, num_epochs):
        train_losses = []  # List to store training losses
        valid_losses = []  # List to store validation losses
        train_accuracies = []  # List to store training accuracies
        valid_accuracies = []  # List to store validation accuracies
        for epoch in range(num_epochs):  # Loop through each epoch
            self.model.train()  # Set the model to training mode
            running_loss = 0.0  # Initialize running loss for training
            correct_train = 0  # Initialize the count of correctly predicted training samples
            total_train = 0  # Initialize the total count of training samples
            for images, labels in train_loader:  # Iterate over training data
                images, labels = images.to(self.device), labels.to(self.device)  # Move data to device
                self.optimizer.zero_grad()  # Zero the parameter gradients
                outputs = self.model(images)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Optimize
                running_loss += loss.item()  # Accumulate the training loss
                _, predicted_train = torch.max(outputs, 1)  # Get the predicted labels
                total_train += labels.size(0)  # Accumulate the total count of training samples
                correct_train += (predicted_train == labels).sum().item()  # Count correct predictions

            train_loss = running_loss / len(train_loader)  # Calculate average training loss
            train_losses.append(train_loss)  # Append training loss to list
            train_accuracy = 100 * correct_train / total_train  # Calculate training accuracy
            train_accuracies.append(train_accuracy)  # Append training accuracy to list

            # Validation
            self.model.eval()  # Set the model to evaluation mode
            valid_loss = 0.0  # Initialize validation loss
            correct_valid = 0  # Initialize the count of correctly predicted validation samples
            total_valid = 0  # Initialize the total count of validation samples
            with torch.no_grad():  # No gradient computation during validation
                for images, labels in valid_loader:  # Iterate over validation data
                    images, labels = images.to(self.device), labels.to(self.device)  # Move data to device
                    outputs = self.model(images)  # Forward pass
                    loss = self.criterion(outputs, labels)  # Compute the loss
                    valid_loss += loss.item()  # Accumulate the validation loss
                    _, predicted_valid = torch.max(outputs, 1)  # Get the predicted labels
                    total_valid += labels.size(0)  # Accumulate the total count of validation samples
                    correct_valid += (predicted_valid == labels).sum().item()  # Count correct predictions

            valid_loss = valid_loss / len(valid_loader)  # Calculate average validation loss
            valid_losses.append(valid_loss)  # Append validation loss to list
            valid_accuracy = 100 * correct_valid / total_valid  # Calculate validation accuracy
            valid_accuracies.append(valid_accuracy)  # Append validation accuracy to list

            # Print epoch-wise training and validation statistics
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

        return train_losses, valid_losses, train_accuracies, valid_accuracies  # Return training and validation statistics
