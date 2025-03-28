import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
import json
from scipy import signal as sp_signal
from skimage.transform import resize

# Import project modules
from models import ConditionalDiscriminator
from dataset import MOD_TYPES, SIGNAL_TYPES
from inference import load_generator, generate_signals

def calculate_mmd(real_samples, fake_samples, kernel='rbf', gamma=1.0):
    """
    Calculates Maximum Mean Discrepancy (MMD) metric
    
    Args:
        real_samples: Real samples
        fake_samples: Generated samples
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel parameter
        
    Returns:
        MMD (lower is better)
    """
    # Flatten samples
    real_flat = real_samples.reshape(real_samples.shape[0], -1)
    fake_flat = fake_samples.reshape(fake_samples.shape[0], -1)
    
    # Function to compute kernel
    def compute_kernel(x, y):
        if kernel == 'rbf':
            # RBF (Gaussian) kernel
            xx = np.dot(x, x.T)
            xy = np.dot(x, y.T)
            yy = np.dot(y, y.T)
            
            vnorm_xx = np.diag(xx).reshape(-1, 1)
            vnorm_xy = np.sum(x**2, axis=1).reshape(-1, 1)
            vnorm_yy = np.diag(yy).reshape(-1, 1)
            
            k_xx = np.exp(-gamma * (vnorm_xx + vnorm_xx.T - 2*xx))
            k_xy = np.exp(-gamma * (vnorm_xy + vnorm_xy.T - 2*xy))
            k_yy = np.exp(-gamma * (vnorm_yy + vnorm_yy.T - 2*yy))
        else:
            # Linear kernel
            k_xx = np.dot(x, x.T)
            k_xy = np.dot(x, y.T)
            k_yy = np.dot(y, y.T)
        
        return k_xx, k_xy, k_yy
    
    # Calculate kernel matrices
    k_xx, k_xy, k_yy = compute_kernel(real_flat, fake_flat)
    
    # Calculate MMD
    m = real_flat.shape[0]
    n = fake_flat.shape[0]
    
    mmd = np.sum(k_xx) / (m*m) + np.sum(k_yy) / (n*n) - 2 * np.sum(k_xy) / (m*n)
    
    return mmd

def visualize_distribution(real_signals, fake_signals, output_dir, method='pca'):
    """
    Visualizes distribution of real vs generated signals using dimensionality reduction
    
    Args:
        real_signals: Real signals
        fake_signals: Generated signals
        output_dir: Output directory
        method: Reduction method ('pca' or 't-sne')
    """
    # Flatten signals
    real_flat = real_signals.reshape(real_signals.shape[0], -1)
    fake_flat = fake_signals.reshape(fake_signals.shape[0], -1)
    
    # Combine for dimensionality reduction
    combined = np.vstack([real_flat, fake_flat])
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(combined)
    else:  # t-SNE
        reducer = TSNE(n_components=2, perplexity=30)
        reduced = reducer.fit_transform(combined)
    
    # Split back into real and fake
    real_reduced = reduced[:real_flat.shape[0]]
    fake_reduced = reduced[real_flat.shape[0]:]
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.scatter(real_reduced[:, 0], real_reduced[:, 1], alpha=0.5, label='Real Signals')
    plt.scatter(fake_reduced[:, 0], fake_reduced[:, 1], alpha=0.5, label='Generated Signals')
    plt.legend()
    plt.title(f'Signal Distribution Visualization using {method.upper()}')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'distribution_{method}.png'))
    plt.close()

def evaluate_by_modulation(generator, dataloader, device, output_dir):
    """
    Evaluates the generator for each modulation type separately
    
    Args:
        generator: Generator model
        dataloader: DataLoader with real data
        device: Device
        output_dir: Directory for output
    """
    # Create directory for results
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    metrics = {mod_name: {'mmd': [], 'w_distance': []} for mod_name in MOD_TYPES.keys()}
    
    # For each modulation type
    for mod_name, mod_idx in MOD_TYPES.items():
        print(f"Evaluating modulation type: {mod_name}")
        
        # Collect real samples of this modulation type
        real_samples = []
        real_sig_types = []
        
        for batch in dataloader:
            signals = batch['signal']
            mod_types = batch['mod_type']
            sig_types = batch['sig_type']
            
            # Filter by modulation type
            mask = (mod_types == mod_idx)
            if mask.sum() > 0:
                real_samples.append(signals[mask])
                real_sig_types.append(sig_types[mask])
        
        if not real_samples:
            print(f"No samples found for modulation type {mod_name}")
            continue
        
        # Concatenate real samples
        real_samples = torch.cat(real_samples, dim=0)
        real_sig_types = torch.cat(real_sig_types, dim=0)
        
        # Generate fake samples
        num_real = real_samples.size(0)
        fake_samples, _, _ = generate_signals(
            generator,
            num_samples=num_real,
            device=device,
            specific_config={'mod_type': mod_idx}
        )
        
        # Convert to numpy
        real_np = real_samples.cpu().numpy()
        fake_np = fake_samples.cpu().numpy()
        
        # Calculate metrics
        mmd = calculate_mmd(real_np, fake_np)
        
        # Calculate Wasserstein distance for each component and average
        w_distances = []
        for i in range(real_np.shape[1]):  # For each channel (I/Q)
            for j in range(real_np.shape[2]):  # For each time point
                w_dist = wasserstein_distance(real_np[:, i, j], fake_np[:, i, j])
                w_distances.append(w_dist)
        avg_w_distance = np.mean(w_distances)
        
        # Save metrics
        metrics[mod_name]['mmd'].append(mmd)
        metrics[mod_name]['w_distance'].append(avg_w_distance)
        
        # Visualize distribution
        visualize_distribution(
            real_np,
            fake_np,
            os.path.join(output_dir, f'mod_{mod_name}'),
            method='pca'
        )
        
        # Visualize examples
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        for i in range(min(5, num_real)):
            # Real signal
            axs[0, i].plot(real_np[i, 0], label='I')
            axs[0, i].plot(real_np[i, 1], label='Q')
            axs[0, i].set_title(f'Real {mod_name}')
            axs[0, i].legend()
            
            # Fake signal
            axs[1, i].plot(fake_np[i, 0], label='I')
            axs[1, i].plot(fake_np[i, 1], label='Q')
            axs[1, i].set_title(f'Generated {mod_name}')
            axs[1, i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'examples_{mod_name}.png'))
        plt.close()
    
    # Metrics summary
    summary = {}
    for mod_name in metrics:
        if metrics[mod_name]['mmd']:
            summary[mod_name] = {
                'mmd': np.mean(metrics[mod_name]['mmd']),
                'w_distance': np.mean(metrics[mod_name]['w_distance'])
            }
    
    # Save metrics
    metrics_df = pd.DataFrame(
        [(mod, data['mmd'], data['w_distance']) for mod, data in summary.items()],
        columns=['Modulation', 'MMD', 'Wasserstein Distance']
    )
    
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_by_modulation.csv'), index=False)
    
    # Visualize metrics
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Modulation', y='MMD', data=metrics_df)
    plt.title('Maximum Mean Discrepancy by Modulation Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mmd_by_modulation.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Modulation', y='Wasserstein Distance', data=metrics_df)
    plt.title('Wasserstein Distance by Modulation Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wasserstein_by_modulation.png'))
    plt.close()
    
    return metrics_df

def evaluate_adversarial_discriminator(generator, discriminator, dataloader, device, output_dir):
    """
    Evaluates the discriminator's ability to distinguish real and generated signals
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataloader: DataLoader with real data
        device: Device
        output_dir: Directory for output
    """
    # Create directory for results
    os.makedirs(output_dir, exist_ok=True)
    
    # Set models to evaluation mode
    generator.eval()
    discriminator.eval()
    
    # Initialize lists to store results
    real_scores = []
    fake_scores = []
    
    # Evaluate real and generated signals
    with torch.no_grad():
        for batch in dataloader:
            real_signals = batch['signal'].to(device)
            mod_types = batch['mod_type'].to(device)
            sig_types = batch['sig_type'].to(device)
            batch_size = real_signals.size(0)
            
            # Evaluate real signals
            real_pred = discriminator(real_signals, mod_types, sig_types)
            real_scores.append(real_pred.cpu().numpy())
            
            # Generate and evaluate fake signals
            z = torch.randn(batch_size, generator.noise_dim, device=device)
            fake_signals = generator(z, mod_types, sig_types)
            fake_pred = discriminator(fake_signals, mod_types, sig_types)
            fake_scores.append(fake_pred.cpu().numpy())
    
    # Concatenate results
    real_scores = np.concatenate(real_scores).flatten()
    fake_scores = np.concatenate(fake_scores).flatten()
    
    # Calculate classification metrics
    threshold = 0.5
    real_correct = np.sum(real_scores >= threshold)
    fake_correct = np.sum(fake_scores < threshold)
    real_accuracy = real_correct / len(real_scores)
    fake_accuracy = fake_correct / len(fake_scores)
    overall_accuracy = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
    
    # Save metrics
    metrics = {
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'overall_accuracy': overall_accuracy
    }
    
    with open(os.path.join(output_dir, 'discriminator_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Visualize score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(real_scores, kde=True, label='Real Signals', alpha=0.6)
    sns.histplot(fake_scores, kde=True, label='Generated Signals', alpha=0.6)
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Discriminator Score')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Discriminator Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discriminator_scores.png'))
    plt.close()
    
    return metrics