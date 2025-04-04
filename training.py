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

def gradient_penalty(real_samples, fake_samples, discriminator, device, mod_types, sig_types):
    """
    Calculates gradient penalty for improving training stability
    """
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)
    
    interpolated_scores = discriminator(interpolated, mod_types, sig_types)
    
    grad_outputs = torch.ones(interpolated_scores.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty_loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty_loss

def visualize_signals(signals, epoch, output_dir, mod_types, sig_types):
    """
    Visualizes generated signals during training
    """
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    signals = signals.cpu().numpy()
    mod_types = mod_types.cpu().numpy()
    sig_types = sig_types.cpu().numpy()
    
    mod_names = {v: k for k, v in MOD_TYPES.items()}
    sig_names = {v: k for k, v in SIGNAL_TYPES.items()}
    
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    
    for i, ax in enumerate(axs.flat):
        if i < len(signals):
            I = signals[i, 0, :]
            Q = signals[i, 1, :]
            
            ax.plot(I, label='I')
            ax.plot(Q, label='Q')
            
            mod_name = mod_names[mod_types[i]]
            sig_name = sig_names[sig_types[i]]
            ax.set_title(f"{mod_name}, {sig_name}")
            
            ax.legend()
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(samples_dir, f'generated_signals_epoch_{epoch}.png'))
    plt.close()

def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, 
                    epoch, g_losses, d_losses, output_dir, final=False):
    """
    Saves training checkpoint
    """
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    filename = 'final_model.pth' if final else f'checkpoint_epoch_{epoch}.pth'
    path = os.path.join(checkpoints_dir, filename)
    
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
    Saves and visualizes loss history
    """
    df = pd.DataFrame({
        'G_loss': g_losses,
        'D_loss': d_losses,
        'Real_Score': real_scores,
        'Fake_Score': fake_scores
    })
    
    csv_path = os.path.join(output_dir, 'loss_history.csv')
    df.to_csv(csv_path, index_label='step')
    print(f"Loss history saved to {csv_path}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator')
    plt.plot(d_losses, label='Discriminator')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
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
    Main training function for Radar Signal GAN
    """
    device = config.device
    print(f"Using device: {device}")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Crear o cargar el escritor de TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    if config.use_synthetic:
        print("Loading synthetic dataset...")
        dataloader = get_synthetic_dataloader(
            batch_size=config.batch_size,
            num_samples=10000,
            signal_length=config.signal_length,
            shuffle=True,
            num_workers=config.num_workers
        )
    else:
        dataloader = get_dataloader(
            hdf5_file=config.data_file,
            batch_size=config.batch_size,
            signal_length=config.signal_length,
            normalize=True,
            shuffle=True,
            num_workers=2
        )
    
    generator = ConditionalGenerator(
        noise_dim=config.noise_dim,
        signal_length=config.signal_length,
        num_channels=2,
        num_mod_types=len(MOD_TYPES),
        num_sig_types=len(SIGNAL_TYPES)
    ).to(device)
    
    discriminator = ConditionalDiscriminator(
        signal_length=config.signal_length,
        num_channels=2,
        num_mod_types=len(MOD_TYPES),
        num_sig_types=len(SIGNAL_TYPES)
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    
    # Inicialización de variables para seguimiento del entrenamiento
    G_losses, D_losses = [], []
    real_scores, fake_scores = [], []
    start_epoch = 0
    global_step = 0
    
    # Cargar checkpoint si existe y si se especifica
    if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint:
        checkpoint_path = config.resume_from_checkpoint
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Cargar estado del modelo
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            # Cargar estado de los optimizadores
            optimizer_G.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_d_state_dict'])
            
            # Cargar historial de pérdidas
            G_losses = checkpoint['g_losses']
            D_losses = checkpoint['d_losses']
            
            # Establecer la época de inicio para continuar
            start_epoch = checkpoint['epoch']
            
            # Calcular el paso global (aproximado si no está guardado)
            global_step = start_epoch * len(dataloader)
            
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")
    
    adversarial_loss = nn.BCELoss()
    
    fixed_noise = torch.randn(16, config.noise_dim, device=device)
    fixed_mod_types = torch.tensor([i % len(MOD_TYPES) for i in range(16)], device=device)
    fixed_sig_types = torch.tensor([i // len(MOD_TYPES) for i in range(16)], device=device)
    
    n_critic = 1
    gp_lambda = 10
    
    print(f"Starting training from epoch {start_epoch + 1} to {config.num_epochs}")
    total_steps = len(dataloader) * (config.num_epochs - start_epoch)
    start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            real_signals = batch['signal'].to(device)
            mod_types = batch['mod_type'].to(device)
            sig_types = batch['sig_type'].to(device)
            batch_size = real_signals.size(0)
            
            real_label = torch.ones(batch_size, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)
            
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, config.noise_dim, device=device)
            
            gen_signals = generator(z, mod_types, sig_types)
            
            g_validity = discriminator(gen_signals, mod_types, sig_types)
            g_loss = adversarial_loss(g_validity, real_label)
            
            g_loss.backward()
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            
            real_pred = discriminator(real_signals, mod_types, sig_types)
            fake_pred = discriminator(gen_signals.detach(), mod_types, sig_types)
            
            d_real_loss = adversarial_loss(real_pred, real_label)
            d_fake_loss = adversarial_loss(fake_pred, fake_label)
            
            gp = gradient_penalty(
                real_signals, 
                gen_signals.detach(), 
                discriminator, 
                device, 
                mod_types, 
                sig_types
            )
            
            d_loss = (d_real_loss + d_fake_loss) / 2 + gp_lambda * gp
            
            d_loss.backward()
            optimizer_D.step()
            
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            real_scores.append(real_pred.mean().item())
            fake_scores.append(fake_pred.mean().item())
            
            writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
            writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
            writer.add_scalar('Score/Real', real_pred.mean().item(), global_step)
            writer.add_scalar('Score/Fake', fake_pred.mean().item(), global_step)
            
            global_step += 1
            torch.cuda.empty_cache()
        
        # Guardar checkpoint después de cada época
        with torch.no_grad():
            gen_signals = generator(fixed_noise, fixed_mod_types, fixed_sig_types)
            visualize_signals(gen_signals, epoch+1, config.output_dir, fixed_mod_types, fixed_sig_types)
        
        save_checkpoint(
            generator, discriminator, optimizer_G, optimizer_D,
            epoch+1, G_losses, D_losses, config.output_dir
        )
    
    save_checkpoint(
        generator, discriminator, optimizer_G, optimizer_D,
        config.num_epochs, G_losses, D_losses, config.output_dir, final=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    save_loss_history(G_losses, D_losses, real_scores, fake_scores, config.output_dir)
    
    return generator, discriminator