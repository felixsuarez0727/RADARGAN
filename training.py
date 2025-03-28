import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Import project modules
from models import ConditionalGenerator, ConditionalDiscriminator
from dataset import get_dataloader, MOD_TYPES, SIGNAL_TYPES
from synthetic_dataset import get_synthetic_dataloader

def visualize_signals(signals, epoch, output_dir, mod_types, sig_types):
        """
        Visualizes and saves a set of generated signals
        """
        # Create folder for samples if it doesn't exist
        samples_dir = os.path.join(output_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        # Convert to CPU and numpy
        signals = signals.cpu().numpy()
        mod_types = mod_types.cpu().numpy()
        sig_types = sig_types.cpu().numpy()
        
        # Invert mapping from numbers to names
        mod_names = {v: k for k, v in MOD_TYPES.items()}
        sig_names = {v: k for k, v in SIGNAL_TYPES.items()}
        
        # Create figure
        fig, axs = plt.subplots(4, 4, figsize=(16, 16))
        
        # Plot each signal
        for i, ax in enumerate(axs.flat):
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
                ax.set_title(f"{mod_name}, {sig_name}")
                
                # Add legend
                ax.legend()
                
                # Configure axes
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Amplitude')
        
        # Adjust spacing
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(samples_dir, f'generated_signals_epoch_{epoch}.png'))
        plt.close()

def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, 
                    epoch, g_losses, d_losses, output_dir, final=False):
        """
        Saves a training checkpoint
        """
        # Create folder for checkpoints if it doesn't exist
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Determine filename
        filename = 'final_model.pth' if final else f'checkpoint_epoch_{epoch}.pth'
        path = os.path.join(checkpoints_dir, filename)
        
        # Save state
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'g_losses': g_losses,
            'd_losses': d_losses,
        }, path)
        
        print(f"Checkpoint saved to {path}")

def save_loss_history(g_losses, d_losses, real_scores, fake_scores, output_dir):
        """
        Saves loss history to a CSV file
        """
        df = pd.DataFrame({
            'G_loss': g_losses,
            'D_loss': d_losses,
            'Real_Score': real_scores,
            'Fake_Score': fake_scores
        })
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'loss_history.csv')
        df.to_csv(csv_path, index_label='step')
        print(f"Loss history saved to {csv_path}")
        
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator')
        plt.plot(d_losses, label='Discriminator')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
        plt.close()
        
        # Plot scores
        plt.figure(figsize=(10, 5))
        plt.plot(real_scores, label='Real Signals')
        plt.plot(fake_scores, label='Fake Signals')
        plt.xlabel('Training Steps')
        plt.ylabel('Discriminator Score')
        plt.legend()
        plt.title('Discriminator Scores')
        plt.savefig(os.path.join(output_dir, 'score_plot.png'))
        plt.close()

def train_gan(config):
    """
    Main function for training the GAN
    """
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create folder for saved models
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Configure TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    # Load dataset
    if hasattr(config, 'use_synthetic') and config.use_synthetic:
        print("Loading synthetic dataset...")
        dataloader = get_synthetic_dataloader(
            batch_size=config.batch_size,
            num_samples=10000,  # Number of synthetic samples
            signal_length=config.signal_length,
            shuffle=True,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 4
        )
    else:
        dataloader = get_dataloader(
            hdf5_file=config.data_file,
            batch_size=config.batch_size,
            signal_length=config.signal_length,
            normalize=True,
            shuffle=True,
            num_workers=config.num_workers if hasattr(config, 'num_workers') else 4
        )
    
    # Initialize models
    generator = ConditionalGenerator(
        noise_dim=config.noise_dim,
        signal_length=config.signal_length,
        num_channels=2,  # I and Q
        num_mod_types=len(MOD_TYPES),
        num_sig_types=len(SIGNAL_TYPES)
    ).to(device)
    
    discriminator = ConditionalDiscriminator(
        signal_length=config.signal_length,
        num_channels=2,  # I and Q
        num_mod_types=len(MOD_TYPES),
        num_sig_types=len(SIGNAL_TYPES)
    ).to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1 if hasattr(config, 'beta1') else 0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1 if hasattr(config, 'beta1') else 0.5, 0.999))
    
    # Loss criterion
    adversarial_loss = nn.BCELoss()
    
    # Tracking metrics
    G_losses = []
    D_losses = []
    real_scores = []
    fake_scores = []
    
    # Prepare fixed noise vector for visualization
    fixed_noise = torch.randn(16, config.noise_dim, device=device)
    fixed_mod_types = torch.tensor([i % len(MOD_TYPES) for i in range(16)], device=device)
    fixed_sig_types = torch.tensor([i // len(MOD_TYPES) for i in range(16)], device=device)
    
    # Main training loop
    n_critic = config.n_critic if hasattr(config, 'n_critic') else 1
    log_interval = config.log_interval if hasattr(config, 'log_interval') else 100
    sample_interval = config.sample_interval if hasattr(config, 'sample_interval') else 1
    save_interval = config.save_interval if hasattr(config, 'save_interval') else 10
    
    total_steps = len(dataloader) * config.num_epochs
    global_step = 0
    start_time = time.time()
    
    print("Starting training...")
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")):
            # Get batch of real data
            real_signals = batch['signal'].to(device)
            mod_types = batch['mod_type'].to(device)
            sig_types = batch['sig_type'].to(device)
            batch_size = real_signals.size(0)
            
            # Configure labels for loss
            real_label = torch.ones(batch_size, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # Generate random noise
            z = torch.randn(batch_size, config.noise_dim, device=device)
            
            # Generate fake signals
            gen_signals = generator(z, mod_types, sig_types)
            
            # Calculate generator loss
            validity = discriminator(gen_signals, mod_types, sig_types)
            g_loss = adversarial_loss(validity, real_label)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            if i % n_critic == 0:
                optimizer_D.zero_grad()
                
                # Calculate loss for real signals
                real_pred = discriminator(real_signals, mod_types, sig_types)
                d_real_loss = adversarial_loss(real_pred, real_label)
                
                # Calculate loss for generated signals
                fake_pred = discriminator(gen_signals.detach(), mod_types, sig_types)
                d_fake_loss = adversarial_loss(fake_pred, fake_label)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                d_loss.backward()
                optimizer_D.step()
            
            # Save metrics
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            real_scores.append(real_pred.mean().item())
            fake_scores.append(fake_pred.mean().item())
            
            # Update TensorBoard
            if global_step % log_interval == 0:
                writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
                writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                writer.add_scalar('Score/Real', real_pred.mean().item(), global_step)
                writer.add_scalar('Score/Fake', fake_pred.mean().item(), global_step)
            
            global_step += 1
        
        # Generate and save samples after each epoch
        if (epoch + 1) % sample_interval == 0:
            with torch.no_grad():
                gen_signals = generator(fixed_noise, fixed_mod_types, fixed_sig_types)
                visualize_signals(gen_signals, epoch+1, config.output_dir, fixed_mod_types, fixed_sig_types)
        
        # Save models periodically
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                generator, discriminator, optimizer_G, optimizer_D,
                epoch, G_losses, D_losses, config.output_dir
            )
    
    # Save final model
    save_checkpoint(
        generator, discriminator, optimizer_G, optimizer_D,
        config.num_epochs, G_losses, D_losses, config.output_dir, final=True
    )
    
    # Calculate total training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Save loss history
    save_loss_history(G_losses, D_losses, real_scores, fake_scores, config.output_dir)

    