import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Mapping of modulation names to indices (same as in dataset.py)
MOD_TYPES = {
    'AM-DSB': 0,
    'AM-SSB': 1,
    'ASK': 2,
    'BPSK': 3,
    'FMCW': 4,
    'PULSED': 5
}

# Mapping of signal names to indices (same as in dataset.py)
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

class SyntheticRadarDataset(Dataset):
    """
    Synthetic dataset that generates artificial signals for testing
    """
    def __init__(self, num_samples=1000, signal_length=128, normalize=True):
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.normalize = normalize
        
        # Generate random labels
        self.mod_labels = np.random.randint(0, len(MOD_TYPES), size=num_samples)
        self.sig_labels = np.random.randint(0, len(SIGNAL_TYPES), size=num_samples)
        
        # Generate synthetic data
        self.signals = []
        for i in range(num_samples):
            mod_type = self.mod_labels[i]
            self.signals.append(self._generate_synthetic_signal(mod_type))
        
        print(f"Synthetic dataset created with {num_samples} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get synthetic data
        signal = self.signals[idx]
        
        # Convert to PyTorch tensor
        signal_tensor = torch.from_numpy(signal).float()
        
        # Get labels
        mod_label = self.mod_labels[idx]
        sig_label = self.sig_labels[idx]
        
        return {
            'signal': signal_tensor,
            'mod_type': torch.tensor(mod_label, dtype=torch.long),
            'sig_type': torch.tensor(sig_label, dtype=torch.long),
            'key': f"synthetic_{idx}"
        }
    
    def _generate_synthetic_signal(self, mod_type):
        """
        Generates a synthetic signal based on modulation type
        """
        # Base frequency and time
        t = np.linspace(0, 1, self.signal_length)
        
        if mod_type == 0:  # AM-DSB
            # Basic AM-DSB signal
            carrier = np.sin(2 * np.pi * 10 * t)
            message = np.sin(2 * np.pi * 1 * t)
            i_data = (1 + 0.5 * message) * carrier
            q_data = np.zeros_like(i_data)
            
        elif mod_type == 1:  # AM-SSB
            # Simplified AM-SSB signal
            carrier = np.sin(2 * np.pi * 10 * t)
            message = np.sin(2 * np.pi * 1 * t)
            i_data = (1 + 0.5 * message) * carrier
            q_data = (1 + 0.5 * message) * np.cos(2 * np.pi * 10 * t)
            
        elif mod_type == 2:  # ASK
            # Amplitude Shift Keying
            carrier = np.sin(2 * np.pi * 10 * t)
            bits = np.random.randint(0, 2, size=8)
            message = np.repeat(bits, self.signal_length // 8)
            i_data = message * carrier[:len(message)]
            q_data = np.zeros_like(i_data)
            
        elif mod_type == 3:  # BPSK
            # Binary Phase Shift Keying
            carrier = np.sin(2 * np.pi * 10 * t)
            bits = 2 * np.random.randint(0, 2, size=8) - 1  # -1 or 1
            message = np.repeat(bits, self.signal_length // 8)
            i_data = message * carrier[:len(message)]
            q_data = np.zeros_like(i_data)
            
        elif mod_type == 4:  # FMCW
            # Frequency Modulated Continuous Wave
            phase = 2 * np.pi * (10 * t + 5 * t**2)
            i_data = np.sin(phase)
            q_data = np.cos(phase)
            
        else:  # PULSED
            # Pulsed signal
            i_data = np.zeros(self.signal_length)
            q_data = np.zeros(self.signal_length)
            
            # Create pulses
            pulse_width = self.signal_length // 8
            pulse_positions = [int(self.signal_length * i / 4) for i in range(4)]
            
            for pos in pulse_positions:
                if pos + pulse_width <= self.signal_length:
                    i_data[pos:pos + pulse_width] = np.sin(2 * np.pi * 20 * t[:pulse_width])
                    q_data[pos:pos + pulse_width] = np.cos(2 * np.pi * 20 * t[:pulse_width])
        
        # Add noise
        i_data = i_data + np.random.normal(0, 0.1, self.signal_length)
        q_data = q_data + np.random.normal(0, 0.1, self.signal_length)
        
        # Normalize if needed
        if self.normalize:
            i_data = (i_data - np.mean(i_data)) / np.std(i_data)
            q_data = (q_data - np.mean(q_data)) / np.std(q_data)
        
        # Combine I and Q into an array of shape [2, signal_length]
        signal = np.stack([i_data, q_data])
        
        return signal

def get_synthetic_dataloader(batch_size=64, num_samples=1000, signal_length=128, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the synthetic dataset
    """
    dataset = SyntheticRadarDataset(
        num_samples=num_samples,
        signal_length=signal_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader