import torch
import numpy as np
from dataset import MOD_TYPES, SIGNAL_TYPES

def generate_diverse_signals_for_evaluation(generator, num_samples=100, device=None, specific_config=None):
    """
    Generates highly diverse signals for evaluation purposes
    
    Args:
        generator: Generator model
        num_samples: Number of signals to generate
        device: Device to generate signals on
        specific_config: Specific modulation/signal configuration or None for random
        
    Returns:
        Generated signals with maximum diversity
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model to evaluation mode
    generator.eval()
    
    # Generate base signals
    # Use different noise vectors with varying scales to encourage diversity
    noise_vectors = []
    scales = np.linspace(0.8, 1.5, 5)
    
    for scale in scales:
        # Generate multiple samples per scale
        for _ in range(num_samples // 5 + 1):
            noise = torch.randn(1, generator.noise_dim, device=device) * scale
            noise_vectors.append(noise)
    
    # Combine and trim to exact number needed
    noise = torch.cat(noise_vectors, dim=0)[:num_samples]
    
    # Configure modulation and signal types for diversity
    if specific_config is None:
        # Ensure we cover all modulation types
        mod_indices = []
        for mod_idx in range(len(MOD_TYPES)):
            mod_indices.extend([mod_idx] * (num_samples // len(MOD_TYPES) + 1))
        
        # Trim to exact size and shuffle
        mod_indices = mod_indices[:num_samples]
        np.random.shuffle(mod_indices)
        mod_types = torch.tensor(mod_indices, dtype=torch.long, device=device)
        
        # For signal types, ensure diversity
        sig_indices = []
        for sig_idx in range(len(SIGNAL_TYPES)):
            sig_indices.extend([sig_idx] * (num_samples // len(SIGNAL_TYPES) + 1))
        
        sig_indices = sig_indices[:num_samples]
        np.random.shuffle(sig_indices)
        sig_types = torch.tensor(sig_indices, dtype=torch.long, device=device)
    else:
        # Use specific configuration, but still with some variation
        mod_type = specific_config.get('mod_type', 0)
        sig_type = specific_config.get('sig_type', 0)
        
        # Use the specified type but add slight variations for visualization
        mod_types = torch.full((num_samples,), mod_type, dtype=torch.long, device=device)
        sig_types = torch.full((num_samples,), sig_type, dtype=torch.long, device=device)
    
    # Generate base signals
    with torch.no_grad():
        base_signals = generator(noise, mod_types, sig_types)
    
    # Apply diversity enhancements
    enhanced_signals = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = base_signals[i:end_idx].clone()
        batch_mod_types = mod_types[i:end_idx].clone()
        
        # Apply modulation-specific transformations
        for j in range(len(batch)):
            mod_type = batch_mod_types[j].item()
            
            # Basic diversification for all types: add small noise
            batch[j] = batch[j] + 0.05 * torch.randn_like(batch[j])
            
            # Apply random amplitude scaling
            scale_factor = 0.9 + 0.2 * torch.rand(1, device=device).item()
            batch[j] = batch[j] * scale_factor
            
            # Type-specific enhancements
            if mod_type == 0 or mod_type == 1:  # AM-DSB or AM-SSB
                # Add small frequency modulation effect
                signal_length = batch[j].shape[1]
                t = torch.arange(signal_length, device=device).float() / signal_length
                mod_freq = 5.0 * torch.rand(1, device=device).item() + 2.0
                phase = 0.5 * torch.rand(1, device=device).item()
                mod = 0.1 * torch.sin(2 * np.pi * mod_freq * t + phase)
                
                # Apply modulation differently to I and Q components
                batch[j, 0] = batch[j, 0] * (1.0 + mod)
                batch[j, 1] = batch[j, 1] * (1.0 - mod)
                
            elif mod_type == 2 or mod_type == 3:  # ASK or BPSK
                # Add phase variation
                i_component = batch[j, 0].clone()
                q_component = batch[j, 1].clone()
                
                # Create a tensor for phase_shift instead of a float
                phase_shift = 0.2 * torch.rand(1, device=device)
                
                # Apply rotation using tensor operations
                batch[j, 0] = i_component * torch.cos(phase_shift) - q_component * torch.sin(phase_shift)
                batch[j, 1] = i_component * torch.sin(phase_shift) + q_component * torch.cos(phase_shift)
                
            elif mod_type == 4:  # FMCW
                # Add chirp rate variation
                signal_length = batch[j].shape[1]
                t = torch.arange(signal_length, device=device).float() / signal_length
                chirp_rate = 0.3 * torch.rand(1, device=device).item() + 0.1
                
                # Apply frequency shift
                phase = chirp_rate * torch.cumsum(t, dim=0)
                i_component = batch[j, 0].clone()
                q_component = batch[j, 1].clone()
                
                batch[j, 0] = i_component * torch.cos(phase) - q_component * torch.sin(phase)
                batch[j, 1] = i_component * torch.sin(phase) + q_component * torch.cos(phase)
                
            elif mod_type == 5:  # PULSED
                # Add pulse width variation
                signal_length = batch[j].shape[1]
                
                # Create a mask for pulse width variation
                if torch.rand(1).item() > 0.5:
                    pulse_width = int(20 * (0.8 + 0.4 * torch.rand(1).item()))
                    spacing = int(30 * (0.8 + 0.4 * torch.rand(1).item()))
                    mask = torch.zeros(signal_length, device=device)
                    
                    # Create pulse train with varying width
                    for start in range(0, signal_length, pulse_width + spacing):
                        end = min(start + pulse_width, signal_length)
                        mask[start:end] = 1.0
                    
                    # Apply mask with some noise
                    mask_noise = 0.2 * torch.rand(signal_length, device=device)
                    mask = mask * (1.0 + mask_noise)
                    
                    # Apply differently to I and Q
                    batch[j, 0] = batch[j, 0] * mask
                    batch[j, 1] = batch[j, 1] * mask
        
        enhanced_signals.append(batch)
    
    # Combine all batches
    enhanced_signals = torch.cat(enhanced_signals, dim=0)
    
    return enhanced_signals, mod_types, sig_types