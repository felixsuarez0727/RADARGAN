import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    """
    Generator for radar signal GAN
    Takes a random noise vector and produces a complex I/Q signal
    """
    def __init__(self, noise_dim=100, signal_length=128, num_channels=2):
        super(Generator, self).__init__()
        
        # Scale factor for transposed convolutional layers
        self.init_size = signal_length // 4
        self.noise_dim = noise_dim
        
        # Initial layer to process noise
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128 * self.init_size),
            nn.BatchNorm1d(128 * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Transposed convolutional layers to generate the signal
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(32, num_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Tanh to normalize values between -1 and 1
        )

    def forward(self, z, mod_type=None, sig_type=None):
        # z is the random noise vector
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size)
        out = self.conv_blocks(out)
        return out

class Discriminator(nn.Module):
    """
    Discriminator for radar signal GAN
    Determines if a signal is real or generated
    """
    def __init__(self, signal_length=128, num_channels=2):
        super(Discriminator, self).__init__()

        # Feature extraction module using CNN
        self.conv_blocks = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool1d(2),
            
            # Second convolutional layer
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool1d(2),
            
            # Third convolutional layer
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool1d(2),
        )
        
        # Calculate size after convolutions and pooling
        ds_size = signal_length // 8
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(128 * ds_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability that the signal is real
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.fc(out)
        return validity

class ConditionalGenerator(nn.Module):
    """
    Conditional generator that can create specific signals based on modulation and signal type
    """
    def __init__(self, noise_dim=100, signal_length=128, num_channels=2, 
                 num_mod_types=6, num_sig_types=8):
        super(ConditionalGenerator, self).__init__()
        
        self.init_size = signal_length // 4
        self.noise_dim = noise_dim
        
        # Embedding layers for categorical conditions
        self.mod_embedding = nn.Embedding(num_mod_types, 20)
        self.sig_embedding = nn.Embedding(num_sig_types, 20)
        
        # Initial layer incorporating noise and conditions
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 40, 128 * self.init_size),  # +40 for embeddings
            nn.BatchNorm1d(128 * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Transposed convolutional layers
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

    def forward(self, z, mod_type, sig_type):
        # Get embeddings for conditions
        mod_embed = self.mod_embedding(mod_type)
        sig_embed = self.sig_embedding(sig_type)
        
        # Concatenate noise and embeddings
        z_condition = torch.cat([z, mod_embed, sig_embed], 1)
        
        # Generate signal
        out = self.fc(z_condition)
        out = out.view(out.shape[0], 128, self.init_size)
        out = self.conv_blocks(out)
        return out

class ConditionalDiscriminator(nn.Module):
    """
    Conditional discriminator that takes into account modulation and signal type
    """
    def __init__(self, signal_length=128, num_channels=2, 
                 num_mod_types=6, num_sig_types=8):
        super(ConditionalDiscriminator, self).__init__()
        
        # Embedding layers for categorical conditions
        self.mod_embedding = nn.Embedding(num_mod_types, 20)
        self.sig_embedding = nn.Embedding(num_sig_types, 20)
        
        # Feature extraction module
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
        
        # Calculate size after convolutions and pooling
        ds_size = signal_length // 8
        
        # Fully connected layers incorporating condition information
        self.fc = nn.Sequential(
            nn.Linear(128 * ds_size + 40, 256),  # +40 for embeddings
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, mod_type, sig_type):
        # Get embeddings for conditions
        mod_embed = self.mod_embedding(mod_type)
        sig_embed = self.sig_embedding(sig_type)
        
        # Extract features from signal
        features = self.conv_blocks(img)
        features = features.view(features.shape[0], -1)
        
        # Concatenate features and embeddings
        features_condition = torch.cat([features, mod_embed, sig_embed], 1)
        
        # Calculate validity
        validity = self.fc(features_condition)
        return validity