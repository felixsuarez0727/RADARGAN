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
                    epoch, g_losses, d_losses, real_scores, fake_scores, output_dir, final=False):
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
        'real_scores': real_scores,
        'fake_scores': fake_scores,
    }, path)
    
    print(f"Checkpoint saved to {path}")

def save_loss_history(g_losses, d_losses, real_scores, fake_scores, output_dir):
    """
    Saves and visualizes loss history with improved plotting
    """
    # Para el DataFrame, asegurar que todos los arrays tengan la misma longitud
    min_length = min(len(g_losses), len(d_losses), len(real_scores), len(fake_scores))
    
    # Solo para el CSV, truncar a la misma longitud
    df = pd.DataFrame({
        'G_loss': g_losses[:min_length],
        'D_loss': d_losses[:min_length],
        'Real_Score': real_scores[:min_length],
        'Fake_Score': fake_scores[:min_length]
    })
    
    csv_path = os.path.join(output_dir, 'loss_history.csv')
    df.to_csv(csv_path, index_label='step')
    print(f"Loss history saved to {csv_path}")
    
    # Para las gráficas NO truncar, usar los arrays completos
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(g_losses)), g_losses, label='Generator')
    plt.plot(range(len(d_losses)), d_losses, label='Discriminator')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'), dpi=150)
    plt.close()
    
    # CORREGIR: Graficar scores del discriminador con valores correctos
    plt.figure(figsize=(12, 6))
    
    # Asegurarse de que los valores no sean None o NaN
    real_scores_clean = [x for x in real_scores if x is not None and not np.isnan(x)]
    fake_scores_clean = [x for x in fake_scores if x is not None and not np.isnan(x)]
    
    if len(real_scores_clean) > 0 and len(fake_scores_clean) > 0:
        plt.plot(range(len(real_scores_clean)), real_scores_clean, label='Real Signals')
        plt.plot(range(len(fake_scores_clean)), fake_scores_clean, label='Fake Signals')
        plt.xlabel('Training Steps')
        plt.ylabel('Discriminator Score')
        plt.legend()
        plt.title('Discriminator Scores')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'score_plot.png'), dpi=150)
    else:
        print("WARNING: No valid discriminator scores to plot")
    plt.close()
    
    # Graficar también las últimas 1000 iteraciones para ver el comportamiento reciente
    plt.figure(figsize=(12, 6))
    
    # Solo mostrar las últimas 1000 iteraciones (o menos si hay menos datos)
    last_n = 1000
    g_tail = g_losses[-last_n:] if len(g_losses) > last_n else g_losses
    d_tail = d_losses[-last_n:] if len(d_losses) > last_n else d_losses
    
    plt.plot(range(len(g_tail)), g_tail, label='Generator')
    plt.plot(range(len(d_tail)), d_tail, label='Discriminator')
    plt.xlabel('Training Steps (últimas iteraciones)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses (Últimas Iteraciones)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_plot_recent.png'), dpi=150)
    plt.close()
    
    # CORREGIR: Gráfica de scores recientes
    plt.figure(figsize=(12, 6))
    
    # Limpiar valores
    real_scores_clean = [x for x in real_scores if x is not None and not np.isnan(x)]
    fake_scores_clean = [x for x in fake_scores if x is not None and not np.isnan(x)]
    
    if len(real_scores_clean) > 0 and len(fake_scores_clean) > 0:
        real_tail = real_scores_clean[-last_n:] if len(real_scores_clean) > last_n else real_scores_clean
        fake_tail = fake_scores_clean[-last_n:] if len(fake_scores_clean) > last_n else fake_scores_clean
        
        plt.plot(range(len(real_tail)), real_tail, label='Real Signals')
        plt.plot(range(len(fake_tail)), fake_tail, label='Fake Signals')
        
        # Intentar ajustar el rango del eje y para mejor visualización
        min_score = min(min(real_tail) if real_tail else 1, min(fake_tail) if fake_tail else 1)
        max_score = max(max(real_tail) if real_tail else 0, max(fake_tail) if fake_tail else 0)
        margin = (max_score - min_score) * 0.1  # 10% de margen
        plt.ylim(min_score - margin, max_score + margin)
        
        plt.xlabel('Training Steps (últimas iteraciones)')
        plt.ylabel('Discriminator Score')
        plt.legend()
        plt.title('Discriminator Scores (Últimas Iteraciones)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'score_plot_recent.png'), dpi=150)
    else:
        print("WARNING: No valid recent discriminator scores to plot")
    plt.close()
    
    # DEPURACIÓN: Imprimir información sobre los arrays para diagnosticar problemas
    print("Longitudes de los arrays:")
    print(f"G_losses: {len(g_losses)}, D_losses: {len(d_losses)}")
    print(f"real_scores: {len(real_scores)}, fake_scores: {len(fake_scores)}")
    
    if len(real_scores) > 0 and len(fake_scores) > 0:
        print("Valores estadísticos:")
        real_np = np.array(real_scores_clean)
        fake_np = np.array(fake_scores_clean)
        print(f"real_scores - min: {np.min(real_np):.4f}, max: {np.max(real_np):.4f}, mean: {np.mean(real_np):.4f}")
        print(f"fake_scores - min: {np.min(fake_np):.4f}, max: {np.max(fake_np):.4f}, mean: {np.mean(fake_np):.4f}")
        
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
    
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    # Añadir estos parámetros para controlar el balance de entrenamiento
    if hasattr(config, 'n_critic'):
        n_critic = config.n_critic
    else:
        n_critic = 5  # Entrenar el discriminador 5 veces por cada actualización del generador
    
    if hasattr(config, 'gp_lambda'):
        gp_lambda = config.gp_lambda
    else:
        gp_lambda = 15.0  # Aumentar la penalización del gradiente
    
    # Configurar Learning Rates diferentes para generador y discriminador
    if hasattr(config, 'lr_g') and config.lr_g is not None:
        lr_g = config.lr_g
    else:
        lr_g = config.lr * 0.5  # Menor tasa para el generador para evitar que aprenda demasiado rápido
    
    if hasattr(config, 'lr_d') and config.lr_d is not None:
        lr_d = config.lr_d
    else:
        lr_d = config.lr  # Mantener tasa original para el discriminador
    
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
    
    # SOLUCIÓN RÁPIDA: Cargar automáticamente desde el checkpoint final anterior
    checkpoint_path = os.path.join(config.output_dir, 'checkpoints', 'final_model.pth')
    G_losses, D_losses = [], []
    real_scores, fake_scores = [], []
    start_epoch = 0
    global_step = 0
    
    if os.path.isfile(checkpoint_path):
        print(f"Cargando automáticamente el checkpoint desde {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Cargar estado del modelo
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Cargar historial de pérdidas si está disponible
        if 'g_losses' in checkpoint:
            G_losses = checkpoint['g_losses']
        if 'd_losses' in checkpoint:
            D_losses = checkpoint['d_losses']
        if 'real_scores' in checkpoint:
            real_scores = checkpoint['real_scores']
        if 'fake_scores' in checkpoint:
            fake_scores = checkpoint['fake_scores']
        
        # Establecer la época de inicio para continuar
        start_epoch = checkpoint['epoch']
        
        # Calcular el paso global (aproximado si no está guardado)
        global_step = start_epoch * len(dataloader)
        
        print(f"Continuando entrenamiento desde la época {start_epoch}")
    else:
        print("No se encontró checkpoint para cargar, comenzando desde cero")
        start_epoch = 0
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    
    # Usar tasas de aprendizaje diferentes
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(config.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(config.beta1, 0.999))
    
    # Cargar optimizadores si está disponible
    if os.path.isfile(checkpoint_path) and 'optimizer_g_state_dict' in checkpoint and 'optimizer_d_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_d_state_dict'])
        print("Optimizadores cargados con éxito")
    
    adversarial_loss = nn.BCELoss()
    
    fixed_noise = torch.randn(16, config.noise_dim, device=device)
    fixed_mod_types = torch.tensor([i % len(MOD_TYPES) for i in range(16)], device=device)
    fixed_sig_types = torch.tensor([i // len(MOD_TYPES) for i in range(16)], device=device)
    
    print(f"Starting training from epoch {start_epoch + 1} to {config.num_epochs}")
    total_steps = len(dataloader) * (config.num_epochs - start_epoch)
    start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")):
            real_signals = batch['signal'].to(device)
            mod_types = batch['mod_type'].to(device)
            sig_types = batch['sig_type'].to(device)
            batch_size = real_signals.size(0)
            
            real_label = torch.ones(batch_size, 1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)
            
            # ---------------------
            # Entrenar Discriminador
            # ---------------------
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                
                # Generar ruido aleatorio con varianza adaptativa
                z_std = 1.0 + 0.1 * np.random.random()  # Varianza variable para diversidad
                z = torch.randn(batch_size, config.noise_dim, device=device) * z_std
                
                # Añadir ruido a las etiquetas reales para label smoothing
                real_smooth = real_label - 0.1 * torch.rand_like(real_label)
                
                # Evaluar discriminador en datos reales
                real_pred = discriminator(real_signals, mod_types, sig_types)
                d_real_loss = adversarial_loss(real_pred, real_smooth)
                
                # Generar señales falsas
                with torch.no_grad():
                    gen_signals = generator(z, mod_types, sig_types)
                
                # Añadir pequeño ruido a las señales generadas para mejorar estabilidad
                noise_level = 0.05
                gen_signals_noisy = gen_signals + noise_level * torch.randn_like(gen_signals)
                
                # Evaluar discriminador en datos generados
                fake_pred = discriminator(gen_signals_noisy.detach(), mod_types, sig_types)
                d_fake_loss = adversarial_loss(fake_pred, fake_label)
                
                # Penalización del gradiente para WGAN-GP
                gp = gradient_penalty(
                    real_signals, 
                    gen_signals.detach(), 
                    discriminator, 
                    device, 
                    mod_types, 
                    sig_types
                )
                
                # Pérdida total del discriminador
                d_loss = d_real_loss + d_fake_loss + gp_lambda * gp
                
                d_loss.backward()
                optimizer_D.step()
            
            # -----------------
            # Entrenar Generador
            # -----------------
            optimizer_G.zero_grad()
            
            # Generar nuevo ruido para el generador
            z = torch.randn(batch_size, config.noise_dim, device=device)
            
            # Generar señales
            gen_signals = generator(z, mod_types, sig_types)
            
            # Calcular pérdida del generador
            g_validity = discriminator(gen_signals, mod_types, sig_types)
            
            # Feature matching loss (tomando características intermedias del discriminador)
            try:
                real_features = discriminator.features(real_signals, mod_types, sig_types)
                fake_features = discriminator.features(gen_signals, mod_types, sig_types)
                feature_loss = torch.mean(torch.abs(real_features.mean(0) - fake_features.mean(0)))
                g_loss = adversarial_loss(g_validity, real_label) + 0.1 * feature_loss
            except (AttributeError, TypeError):
                # Si el discriminador no tiene la función 'features' o hay un error
                g_loss = adversarial_loss(g_validity, real_label)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Registrar valores de pérdida y puntuaciones
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            real_scores.append(real_pred.mean().item())
            fake_scores.append(fake_pred.mean().item())
            
            # Logging a TensorBoard
            writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
            writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
            writer.add_scalar('Score/Real', real_pred.mean().item(), global_step)
            writer.add_scalar('Score/Fake', fake_pred.mean().item(), global_step)
            
            global_step += 1
            
            # Limpiar memoria
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                gen_signals = generator(fixed_noise, fixed_mod_types, fixed_sig_types)
                visualize_signals(gen_signals, epoch+1, config.output_dir, fixed_mod_types, fixed_sig_types)
            
            save_checkpoint(
                generator, discriminator, optimizer_G, optimizer_D,
                epoch+1, G_losses, D_losses, real_scores, fake_scores, config.output_dir
            )
        
        # Guardar visualización después de cada época
        with torch.no_grad():
            gen_signals = generator(fixed_noise, fixed_mod_types, fixed_sig_types)
            visualize_signals(gen_signals, epoch+1, config.output_dir, fixed_mod_types, fixed_sig_types)
    
    save_checkpoint(
        generator, discriminator, optimizer_G, optimizer_D,
        config.num_epochs, G_losses, D_losses, real_scores, fake_scores, config.output_dir, final=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    save_loss_history(G_losses, D_losses, real_scores, fake_scores, config.output_dir)
    
    return generator, discriminator