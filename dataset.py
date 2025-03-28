import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Mapping of modulation names to indices
MOD_TYPES = {
    'AM-DSB': 0,
    'AM-SSB': 1,
    'ASK': 2,
    'BPSK': 3,
    'FMCW': 4,
    'PULSED': 5
}

# Mapping of signal names to indices
SIGNAL_TYPES = {
    'AM radio': 0,
    'Short-range': 1,
    'Satellite Communication': 2,
    'Radar Altimeter': 3,
    'Air-Ground-MTI': 4,
    'Airborne-detection': 5,
    'Airborne-range': 6,
    'Ground mapping': 7
}

class RadarSignalDataset(Dataset):
    """
    Dataset for loading radar/communication signals from the HDF5 file
    """
    def __init__(self, hdf5_file, signal_length=128, normalize=True, transform=None):
        """
        Args:
            hdf5_file (str): Path to HDF5 file with signals
            signal_length (int): Length of signal to process (will be truncated or padded)
            normalize (bool): Whether to normalize the signals
            transform (callable, optional): Additional transformations
        """
        self.file_path = hdf5_file
        self.signal_length = signal_length
        self.normalize = normalize
        self.transform = transform
        self.keys = []
        self.mod_labels = []
        self.sig_labels = []
        
        # Load keys and labels from HDF5 file
        with h5py.File(self.file_path, 'r') as f:
            # Store all keys and their corresponding labels
            for key in f.keys():
                # Expected format: ('MOD_TYPE', 'SIGNAL_TYPE', SNR, NUM)
                # Extract information from the key
                key_parts = key.strip("()").replace("'", "").split(", ")
                if len(key_parts) >= 2:
                    mod_type = key_parts[0]
                    signal_type = key_parts[1]
                    
                    # Only add if types are in our mappings
                    if mod_type in MOD_TYPES and signal_type in SIGNAL_TYPES:
                        self.keys.append(key)
                        self.mod_labels.append(MOD_TYPES[mod_type])
                        self.sig_labels.append(SIGNAL_TYPES[signal_type])
        
        # Convert lists to numpy arrays for efficiency
        self.mod_labels = np.array(self.mod_labels)
        self.sig_labels = np.array(self.sig_labels)
        
        print(f"Dataset loaded with {len(self.keys)} samples")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            # Get signal from HDF5 file
            key = self.keys[idx]
            data = f[key][()]
            
            # Process signal based on its format
            if data.ndim == 1:
                # If it's a 1D array, assume first half is I and second half is Q
                mid = len(data) // 2
                i_data = data[:mid]
                q_data = data[mid:]
            elif data.ndim == 2 and data.shape[0] >= 2:
                # If it's a 2D array, assume first row is I and second row is Q
                i_data = data[0, :]
                q_data = data[1, :]
            else:
                raise ValueError(f"Unexpected data format for key {key}: shape {data.shape}")
            
            # Adjust signal length
            i_data = self._adjust_length(i_data)
            q_data = self._adjust_length(q_data)
            
            # Normalize if needed
            if self.normalize:
                i_mean, i_std = np.mean(i_data), np.std(i_data)
                q_mean, q_std = np.mean(q_data), np.std(q_data)
                
                i_data = (i_data - i_mean) / (i_std if i_std > 0 else 1)
                q_data = (q_data - q_mean) / (q_std if q_std > 0 else 1)
            
            # Combine I and Q into a tensor of shape [2, signal_length]
            signal = np.stack([i_data, q_data])
            
            # Convert to PyTorch tensor
            signal_tensor = torch.from_numpy(signal).float()
            
            # Apply additional transformations if any
            if self.transform:
                signal_tensor = self.transform(signal_tensor)
            
            # Get labels
            mod_label = self.mod_labels[idx]
            sig_label = self.sig_labels[idx]
            
            return {
                'signal': signal_tensor,
                'mod_type': torch.tensor(mod_label, dtype=torch.long),
                'sig_type': torch.tensor(sig_label, dtype=torch.long),
                'key': key
            }
    
    def _adjust_length(self, data):
        """Adjusts the signal length to the required size"""
        if len(data) > self.signal_length:
            # Truncate
            return data[:self.signal_length]
        elif len(data) < self.signal_length:
            # Pad with zeros
            padded = np.zeros(self.signal_length)
            padded[:len(data)] = data
            return padded
        else:
            return data

def get_dataloader(hdf5_file, batch_size=64, signal_length=128, normalize=True, 
                  shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the radar signal dataset
    
    Args:
        hdf5_file (str): Path to HDF5 file
        batch_size (int): Batch size
        signal_length (int): Signal length
        normalize (bool): Whether to normalize signals
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of workers for loading data
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    dataset = RadarSignalDataset(
        hdf5_file=hdf5_file,
        signal_length=signal_length,
        normalize=normalize
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader