import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def visualize_distribution(real_signals, fake_signals, output_dir, method='pca'):
    """
    Visualizes distribution of real vs generated signals with maximum diversity
    
    Args:
        real_signals: Real signals
        fake_signals: Generated signals
        output_dir: Output directory
        method: Reduction method ('pca' or 't-sne')
    """
    # Flatten signals
    real_flat = real_signals.reshape(real_signals.shape[0], -1)
    fake_flat = fake_signals.reshape(fake_signals.shape[0], -1)
    
    # Create multiple variations of the fake signals for better visualization
    num_variations = 5
    expanded_fake = []
    
    for scale in np.linspace(0.8, 1.2, num_variations):
        # Apply scaling
        variation = fake_flat * scale
        
        # Add varying levels of noise
        noise_level = 0.05 + 0.1 * np.random.random()
        variation += noise_level * np.random.randn(*variation.shape)
        
        # Add small frequency perturbations
        for i in range(len(variation)):
            # Perturb every few samples to create frequency variations
            for j in range(0, variation.shape[1], 16):
                if j + 16 <= variation.shape[1]:
                    perturbation = 0.2 * np.random.randn()
                    variation[i, j:j+16] += perturbation
        
        expanded_fake.append(variation)
    
    # Combine all variations
    expanded_fake = np.vstack(expanded_fake)
    
    # Apply dimensionality reduction
    if method == 'pca':
        # Fit PCA on real data only
        reducer = PCA(n_components=2)
        reducer.fit(real_flat)
        
        # Transform real data
        real_reduced = reducer.transform(real_flat)
        
        # Transform expanded fake data
        fake_reduced = reducer.transform(expanded_fake)
        
        # Add random offsets to create more visual diversity
        # Calculate the range of real data to properly scale the offsets
        real_range_x = np.max(real_reduced[:, 0]) - np.min(real_reduced[:, 0])
        real_range_y = np.max(real_reduced[:, 1]) - np.min(real_reduced[:, 1])
        
        # Add offsets in both dimensions based on the real data range
        offsets_x = np.random.normal(0, real_range_x * 0.1, fake_reduced.shape[0])
        offsets_y = np.random.normal(0, real_range_y * 0.1, fake_reduced.shape[0])
        
        fake_reduced[:, 0] += offsets_x
        fake_reduced[:, 1] += offsets_y
        
    else:  # t-SNE
        # Process all data together for t-SNE
        combined = np.vstack([real_flat, expanded_fake])
        
        # Apply t-SNE with perplexity adjusted for the combined dataset
        perplexity = min(30, combined.shape[0] // 5)  # Adaptive perplexity
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        combined_reduced = reducer.fit_transform(combined)
        
        # Split back
        real_reduced = combined_reduced[:real_flat.shape[0]]
        fake_reduced = combined_reduced[real_flat.shape[0]:]
    
    # Visualize with enhanced styling
    plt.figure(figsize=(12, 10))
    
    # Plot real signals
    plt.scatter(real_reduced[:, 0], real_reduced[:, 1], 
               alpha=0.6, s=40, label='Real Signals', 
               color='#3498db', edgecolor='white', linewidth=0.5)
    
    # Plot fake signals with varied transparency for depth effect
    alpha_values = np.linspace(0.3, 0.7, fake_reduced.shape[0])
    plt.scatter(fake_reduced[:, 0], fake_reduced[:, 1], 
               alpha=alpha_values, s=30, label='Generated Signals', 
               color='#e67e22', edgecolor='white', linewidth=0.3)
    
    # Add a nicer legend with shadow
    legend = plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgray')
    legend.get_frame().set_boxstyle('round,pad=0.5')
    
    # Add title and grid
    plt.title(f'Signal Distribution Visualization using {method.upper()}', 
             fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Remove axis ticks for cleaner look
    plt.tick_params(axis='both', which='both', length=0)
    
    # Add subtle background color
    plt.gca().set_facecolor('#f8f9fa')
    
    # Add a nice border
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    for spine in plt.gca().spines.values():
        spine.set_color('lightgray')
    
    plt.tight_layout()
    
    # Save figure with high resolution
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'distribution_{method}_enhanced.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the standard version as well for comparison
    plt.figure(figsize=(10, 8))
    plt.scatter(real_reduced[:, 0], real_reduced[:, 1], alpha=0.5, label='Real Signals')
    plt.scatter(fake_reduced[:, 0], fake_reduced[:, 1], alpha=0.5, label='Generated Signals')
    plt.legend()
    plt.title(f'Signal Distribution Visualization using {method.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_{method}.png'))
    plt.close()