import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import json

# Import project modules
from models import ConditionalGenerator
from dataset import MOD_TYPES, SIGNAL_TYPES

def load_generator(checkpoint_path, device=None):
    """
    Loads a pre-trained generator model
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded generator model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    generator = ConditionalGenerator(
        noise_dim=100,  # Assume noise dimension of 100
        signal_length=128,  # Assume signal length of 128
        num_channels=2,  # I and Q
        num_mod_types=len(MOD_TYPES),
        num_sig_types=len(SIGNAL_TYPES)
    ).to(device)
    
    # Load weights
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # Set to evaluation mode
    generator.eval()
    
    print(f"Generator loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return generator

def generate_signals(generator, num_samples=10, device=None, specific_config=None):
    """
    Generates signals using the trained model
    
    Args:
        generator: Generator model
        num_samples: Number of signals to generate
        device: Device to generate signals on
        specific_config: Specific modulation/signal configuration or None for random
        
    Returns:
        Generated signals, modulation types, and signal types
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model to evaluation mode
    generator.eval()
    
    # Generate random noise
    noise = torch.randn(num_samples, generator.noise_dim, device=device)
    
    # Configure modulation and signal types
    if specific_config is None:
        # Random
        mod_types = torch.randint(0, len(MOD_TYPES), (num_samples,), device=device)
        sig_types = torch.randint(0, len(SIGNAL_TYPES), (num_samples,), device=device)
    else:
        # Specific configuration
        mod_type = specific_config.get('mod_type', 0)
        sig_type = specific_config.get('sig_type', 0)
        mod_types = torch.full((num_samples,), mod_type, dtype=torch.long, device=device)
        sig_types = torch.full((num_samples,), sig_type, dtype=torch.long, device=device)
    
    # Generate signals
    with torch.no_grad():
        generated_signals = generator(noise, mod_types, sig_types)
    
    return generated_signals, mod_types, sig_types

def visualize_generated_signals(signals, mod_types, sig_types, output_path=None):
    """
    Visualizes the generated signals
    
    Args:
        signals: Tensor with generated signals
        mod_types: Modulation types
        sig_types: Signal types
        output_path: Path to save the visualization or None to display it
    """
    # Convert to CPU and numpy
    signals = signals.cpu().numpy()
    mod_types = mod_types.cpu().numpy()
    sig_types = sig_types.cpu().numpy()
    
    # Get name mapping
    mod_names = {v: k for k, v in MOD_TYPES.items()}
    sig_names = {v: k for k, v in SIGNAL_TYPES.items()}
    
    # Calculate figure dimensions
    n_cols = min(5, len(signals))
    n_rows = (len(signals) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axs = axs.flatten() if n_rows * n_cols > 1 else [axs]
    
    # Plot each signal
    for i, ax in enumerate(axs):
        if i < len(signals):
            # Get I/Q components
            I = signals[i, 0, :]
            Q = signals[i, 1, :]
            
            # Plot
            ax.plot(I, label='I')
            ax.plot(Q, label='Q')
            
            # Add title with modulation and signal type
            mod_name = mod_names[mod_types[i]]
            sig_name = sig_names[sig_types[i]]
            ax.set_title(f"{mod_name}\n{sig_name}", fontsize=10)
            
            # Add legend
            ax.legend()
            
            # Configure axes
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')
        else:
            # Hide empty axes
            ax.axis('off')
    
    # Adjust spacing
    plt.tight_layout()
    
    # Save or display figure
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

def export_signals_to_hdf5(signals, mod_types, sig_types, output_file):
    """
    Exports generated signals to an HDF5 file
    
    Args:
        signals: Tensor with generated signals
        mod_types: Modulation types
        sig_types: Signal types
        output_file: Path to HDF5 file
    """
    # Convert to CPU and numpy
    signals = signals.cpu().numpy()
    mod_types = mod_types.cpu().numpy()
    sig_types = sig_types.cpu().numpy()
    
    # Get name mapping
    mod_names = {v: k for k, v in MOD_TYPES.items()}
    sig_names = {v: k for k, v in SIGNAL_TYPES.items()}
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        for i in range(len(signals)):
            # Create key
            mod_name = mod_names[mod_types[i]]
            sig_name = sig_names[sig_types[i]]
            key = f"('{mod_name}', '{sig_name}', 0, {i})"
            
            # Save signal
            f.create_dataset(key, data=signals[i])
    
    print(f"Generated signals exported to {output_file}")