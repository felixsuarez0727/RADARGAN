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

def generate_signals(generator, num_samples=10, device=None, specific_config=None, enhance_diversity=True):
    """
    Generates signals using the trained model with enhanced diversity
    
    Args:
        generator: Generator model
        num_samples: Number of signals to generate
        device: Device to generate signals on
        specific_config: Specific modulation/signal configuration or None for random
        enhance_diversity: Whether to apply diversity enhancement techniques
        
    Returns:
        Generated signals, modulation types, and signal types
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model to evaluation mode
    generator.eval()
    
    # Generate multiple noise vectors with different scales for diversity
    if enhance_diversity:
        noise_vectors = []
        scales = [0.9, 1.0, 1.1, 1.2]
        samples_per_scale = num_samples // len(scales) + 1
        
        for scale in scales:
            noise = torch.randn(samples_per_scale, generator.noise_dim, device=device) * scale
            noise_vectors.append(noise)
        
        # Combine and trim to exact number needed
        noise = torch.cat(noise_vectors, dim=0)[:num_samples]
    else:
        # Standard random noise
        noise = torch.randn(num_samples, generator.noise_dim, device=device)
    
    # Configure modulation and signal types
    if specific_config is None:
        # For diversity, ensure we cover different modulation types
        if enhance_diversity and num_samples >= len(MOD_TYPES):
            # Distribute evenly across modulation types
            mod_indices = []
            for mod_idx in range(len(MOD_TYPES)):
                mod_indices.extend([mod_idx] * (num_samples // len(MOD_TYPES)))
            
            # Add any remaining samples
            mod_indices.extend([0] * (num_samples - len(mod_indices)))
            
            # Shuffle the indices
            np.random.shuffle(mod_indices)
            mod_types = torch.tensor(mod_indices, dtype=torch.long, device=device)
            
            # Randomize signal types
            sig_types = torch.randint(0, len(SIGNAL_TYPES), (num_samples,), device=device)
        else:
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
        
        # Apply post-processing for diversity if requested
        if enhance_diversity:
            # Add small noise to generated signals
            noise_level = 0.05
            noise_tensor = torch.randn_like(generated_signals) * noise_level
            generated_signals = generated_signals + noise_tensor
            
            # Apply individual transformations to each sample
            for i in range(num_samples):
                # Random amplitude scaling (±5%)
                if torch.rand(1).item() > 0.5:
                    scale = 0.95 + 0.1 * torch.rand(1).item()
                    generated_signals[i] = generated_signals[i] * scale
                
                # Small random phase shifts for some samples
                if torch.rand(1).item() > 0.7:
                    # Adjust I/Q relationship slightly
                    i_component = generated_signals[i, 0].clone()
                    q_component = generated_signals[i, 1].clone()
                    alpha = 0.1 * torch.rand(1).item()
                    generated_signals[i, 0] = i_component * (1-alpha) + q_component * alpha
                    generated_signals[i, 1] = q_component * (1-alpha) - i_component * alpha
    
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
    
def generate_ensemble_signals(checkpoint_paths, num_samples=10, device=None, specific_config=None):
    """
    Generate signals using an ensemble of models from different checkpoints
    
    Args:
        checkpoint_paths: List of paths to checkpoint files
        num_samples: Total number of signals to generate
        device: Device to generate signals on
        specific_config: Specific modulation/signal configuration or None for random
        
    Returns:
        Generated signals, modulation types, and signal types
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_signals = []
    all_mod_types = []
    all_sig_types = []
    
    # Calculate how many samples to generate from each checkpoint
    samples_per_checkpoint = max(1, num_samples // len(checkpoint_paths))
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        # For the last checkpoint, adjust to get exactly the requested total
        if i == len(checkpoint_paths) - 1:
            remaining = num_samples - len(all_signals) if all_signals else samples_per_checkpoint
            samples = remaining
        else:
            samples = samples_per_checkpoint
        
        # Skip if no samples needed
        if samples <= 0:
            continue
            
        # Load generator
        generator = load_generator(checkpoint_path, device)
        
        # Generate signals with this checkpoint
        signals, mod_types, sig_types = generate_signals(
            generator, 
            num_samples=samples, 
            device=device, 
            specific_config=specific_config,
            enhance_diversity=True
        )
        
        # Add to collections
        all_signals.append(signals)
        all_mod_types.append(mod_types)
        all_sig_types.append(sig_types)
    
    # Combine results
    combined_signals = torch.cat(all_signals, dim=0)
    combined_mod_types = torch.cat(all_mod_types, dim=0)
    combined_sig_types = torch.cat(all_sig_types, dim=0)
    
    return combined_signals, combined_mod_types, combined_sig_types