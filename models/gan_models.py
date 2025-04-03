import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalGenerator(nn.Module):
    """
    Optimized Conditional Generator for radar signal generation
    """
    def __init__(self, 
                 noise_dim=100, 
                 signal_length=128, 
                 num_channels=2, 
                 num_mod_types=6, 
                 num_sig_types=8):
        super(ConditionalGenerator, self).__init__()
        
        self.init_size = signal_length // 4
        self.noise_dim = noise_dim
        
        # More efficient embedding layers
        self.mod_embedding = nn.Sequential(
            nn.Embedding(num_mod_types, 40),
            nn.Linear(40, 40),
            nn.SiLU()  # Faster activation function
        )
        
        self.sig_embedding = nn.Sequential(
            nn.Embedding(num_sig_types, 40),
            nn.Linear(40, 40),
            nn.SiLU()
        )
        
        # Initial processing with more efficient normalization
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 80, 256 * self.init_size),
            nn.LayerNorm(256 * self.init_size),  # More GPU-friendly normalization
            nn.SiLU(),
            nn.Dropout(0.2)  # Slightly reduced dropout
        )
        
        # Enhanced transposed convolutional layers
        self.conv_blocks = nn.Sequential(
            # First block with spectral normalization
            nn.utils.spectral_norm(nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)),
            nn.GroupNorm(32, 128),  # More efficient normalization
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.utils.spectral_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.utils.spectral_norm(nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            nn.ConvTranspose1d(32, num_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, mod_type, sig_type):
        # Get embeddings for conditions
        mod_embed = self.mod_embedding(mod_type)
        sig_embed = self.sig_embedding(sig_type)
        
        # Concatenate noise and embeddings
        z_condition = torch.cat([z, mod_embed, sig_embed], 1)
        
        # Generate signal
        out = self.fc(z_condition)
        out = out.view(out.shape[0], 256, self.init_size)
        out = self.conv_blocks(out)
        
        return out

class ConditionalDiscriminator(nn.Module):
    """
    Optimized Conditional Discriminator with improved feature extraction
    """
    def __init__(self, 
                 signal_length=128, 
                 num_channels=2, 
                 num_mod_types=6, 
                 num_sig_types=8):
        super(ConditionalDiscriminator, self).__init__()
        
        # Enhanced embedding layers with SiLU activation
        self.mod_embedding = nn.Sequential(
            nn.Embedding(num_mod_types, 40),
            nn.Linear(40, 40),
            nn.SiLU()
        )
        self.sig_embedding = nn.Sequential(
            nn.Embedding(num_sig_types, 40),
            nn.Linear(40, 40),
            nn.SiLU()
        )
        
        # More efficient feature extraction
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(num_channels, 32, kernel_size=3, stride=1, padding=1)),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)),
                nn.GroupNorm(16, 64),
                nn.SiLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)),
                nn.GroupNorm(32, 128),
                nn.SiLU(),
                nn.Dropout(0.2)
            )
        ])
        
        # Adaptive pooling with reduced size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)  # Reduced from 8 to 4
        
        # Optimized classification head
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 + 80, 256),  # Adjusted for reduced pooling
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, mod_type, sig_type):
        # Get embeddings
        mod_embed = self.mod_embedding(mod_type)
        sig_embed = self.sig_embedding(sig_type)
        
        # Feature extraction
        features = img
        for conv_block in self.conv_blocks:
            features = conv_block(features)
        
        # Adaptive pooling
        features = self.adaptive_pool(features)
        features = features.view(features.shape[0], -1)
        
        # Concatenate features and condition embeddings
        features_condition = torch.cat([features, mod_embed, sig_embed], 1)
        
        # Final classification
        validity = self.fc(features_condition)
        return validity