import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset class that handles time-series data for use in neural network training.
    It returns a sequence of concatenated input vectors and the corresponding target AQI value.
    
    Attributes:
        dataframe (pd.DataFrame): The dataframe containing the time-series data.
        seq_length (int): The length of the time-series sequence to be used as input.
        transform (callable, optional): An optional transform to be applied on the samples.
    
    Methods:
        __len__: Returns the total number of sequences in the dataset.
        __getitem__: Retrieves a sequence-target pair from the dataset at the specified index.
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 seq_length: int,
                 transform=None):
        """
        Initializes the TimeSeriesDataset with the given dataframe, sequence length, and an optional transform.
        
        Parameters:
            dataframe (pd.DataFrame): The dataframe containing the time-series data.
            seq_length (int): The length of the time-series sequence to be used as input.
            transform (callable, optional): An optional transform to be applied on the samples.
        """
        self.dataframe = dataframe
        self.seq_length = seq_length
        self.transform = transform

    def __len__(self):
        """Returns the total number of sequences that can be generated from the dataframe."""
        return len(self.dataframe) - self.seq_length + 1

    def __getitem__(self, idx):
        """
        Retrieves a sequence-target pair from the dataset at the specified index.
        
        Parameters:
            idx (int): The index of the sequence to retrieve.
            
        Returns:
            (input,output): A tuple containing the input sequence (as a flattened numpy array) and the target AQI value.
        """
        sequence = self.dataframe.iloc[idx:idx + self.seq_length]
        x = sequence.iloc[:, :-1].to_numpy().reshape(-1)
        y = sequence.iloc[-1, -1]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

class NumpyToTensor:
    """
    A transform class that converts a numpy array to a PyTorch tensor.
    It can handle both input sequences and single-value targets.
    """
    def __call__(self, sample):
        """
        Applies the transform to the given sample.
        
        Parameters:
            sample (np.ndarray or scalar): The input sample, which can be a numpy array or a scalar value.
            
        Returns:
            torch.Tensor: The sample converted to a PyTorch tensor of type float.
        """
        return torch.from_numpy(sample).float() if isinstance(sample, np.ndarray) else torch.tensor(sample).float()