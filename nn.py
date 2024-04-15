import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim as optim


class TimeSeriesNN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim1, 
                 hidden_dim2, 
                 output_dim):
        super(TimeSeriesNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class TimeSeriesTrainer:
    """
    Trainer class for a time-series forecasting neural network model.
    
    This trainer will handle both training and validation phases and will output
    the progress of training using tqdm progress bars.
    
    Attributes:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.modules.loss._Loss): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
    
    Methods:
        train_epoch: Runs a single training epoch.
        validate_epoch: Runs a single validation epoch.
        train: Runs the training process for a specified number of epochs.
    """
    def __init__(self, model, train_loader, val_loader, lr=0.001, device='cpu'):
        """
        Initializes the trainer with the model, data loaders, and other settings. Initializes adam optimizer.
        
        Parameters:
            model (torch.nn.Module): The neural network model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
            lr (float): Learning rate for the optimizer. Default is 0.001.
            device (torch.device): Device where training will take place.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.device = device
        model = model.to(device)

    def train_epoch(self):
        """
        Conducts a single epoch of training over the entire training dataset.
        
        Iterates over the training DataLoader, passing the data through the model,
        calculating loss, and updating the model parameters.
        
        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()  
        running_loss = 0.0
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, targets in loop:
            inputs,targets = inputs.to(self.device),targets.to(self.device)
            self.optimizer.zero_grad()  
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss=loss.cpu().item())
            running_loss += loss.cpu().item()
        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        """
        Conducts a validation run over the entire validation dataset.
        
        Iterates over the validation DataLoader and passes the data through the model
        to compute the loss. The model parameters are not updated.
        
        Returns:
            float: The average validation loss for the epoch.
        """
        self.model.eval()  
        validation_loss = 0.0
        loop = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for inputs, targets in loop:
                inputs,targets = inputs.to(self.device),targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                validation_loss += loss.cpu().item()
                loop.set_postfix(loss=loss.cpu().item())
        return validation_loss / len(self.val_loader)
    
    def train(self, epochs):
        """
        Runs the training and validation loops for a specified number of epochs.

        For each epoch, it trains the model on the training dataset and evaluates it on the validation dataset.
        Prints the average training and validation losses after each epoch.

        Parameters:
            epochs (int): The number of epochs to train the model.
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        print('Finished Training')
