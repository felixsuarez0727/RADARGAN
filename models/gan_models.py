import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Base Generator for radar signal generation
    """
    def __init__(self, noise_dim=100, signal_length=128, num_channels=2):
        super(Generator, self).__init__()
        
        self.init_size = signal_length // 4
        self.noise_dim = noise_dim
        
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128 * self.init_size),
            nn.BatchNorm1d(128 * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(32, num_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size)
        out = self.conv_blocks(out)
        return out

class Discriminator(nn.Module):
    """
    Base Discriminator for radar signal classification
    """
    def __init__(self, signal_length=128, num_channels=2):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool1d(2),
        )
        
        ds_size = signal_length // 8
        
        self.fc = nn.Sequential(
            nn.Linear(128 * ds_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.fc(out)
        return validity

class ConditionalGenerator(nn.Module):
    """
    Optimized Conditional Generator for radar signal generation with improved diversity
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
        self.signal_length = signal_length
        
        # Embedding dimension más grande
        embedding_dim = 64
        
        # Capa de embedding más profunda para modulation type
        self.mod_embedding = nn.Sequential(
            nn.Embedding(num_mod_types, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Capa de embedding más profunda para signal type
        self.sig_embedding = nn.Sequential(
            nn.Embedding(num_sig_types, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Initial processing with more efficient normalization
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 2*embedding_dim, 256 * self.init_size),
            nn.LayerNorm(256 * self.init_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Bloque residual para mejores gradientes
        self.res_block = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256)
        )
        
        # Enhanced transposed convolutional layers
        self.conv_blocks = nn.Sequential(
            # First block with spectral normalization
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)),
            nn.GroupNorm(32, 128),  # More efficient normalization
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.utils.spectral_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.utils.spectral_norm(nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1)),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(32, num_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # Inicializar pesos para mejor convergencia
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, mod_type, sig_type):
        # Get embeddings for conditions
        mod_embed = self.mod_embedding(mod_type)
        sig_embed = self.sig_embedding(sig_type)
        
        # Concatenate noise and embeddings
        z_condition = torch.cat([z, mod_embed, sig_embed], 1)
        
        # Generate signal
        out = self.fc(z_condition)
        out = out.view(out.shape[0], 256, self.init_size)
        
        # Aplicar bloque residual
        residual = out
        out = self.res_block(out)
        out = out + residual  # Conexión residual
        
        # Aplicar bloques convolucionales
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
        
        # Embedding dimension más grande
        embedding_dim = 64
        
        # Enhanced embedding layers with LeakyReLU activation
        self.mod_embedding = nn.Sequential(
            nn.Embedding(num_mod_types, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2)
        )
        self.sig_embedding = nn.Sequential(
            nn.Embedding(num_sig_types, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2)
        )
        
        # More efficient feature extraction
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(num_channels, 32, kernel_size=3, stride=1, padding=1)),
                nn.GroupNorm(8, 32),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)),
                nn.GroupNorm(16, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)),
                nn.GroupNorm(32, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            )
        ])
        
        # Adaptive pooling with reduced size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)  # Reduced from 8 to 4
        
        # Optimized classification head
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 + 2*embedding_dim, 256),  # Adjusted for reduced pooling and larger embeddings
            nn.LeakyReLU(0.2),
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
    
    def features(self, img, mod_type, sig_type):
        """
        Extrae características intermedias para feature matching
        """
        # Get embeddings
        mod_embed = self.mod_embedding(mod_type)
        sig_embed = self.sig_embedding(sig_type)
        
        # Feature extraction
        features = img
        feature_maps = []
        for conv_block in self.conv_blocks:
            features = conv_block(features)
            feature_maps.append(features)
        
        # Adaptive pooling
        features = self.adaptive_pool(features)
        features = features.view(features.shape[0], -1)
        
        # Retornar el tensor de características antes de la capa final
        return features