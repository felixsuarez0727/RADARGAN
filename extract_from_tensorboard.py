import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from models import ConditionalGenerator, ConditionalDiscriminator
from dataset import MOD_TYPES, SIGNAL_TYPES

def extract_tensorboard_data(log_dir):
    """
    Extracts data from TensorBoard event files
    """
    print(f"Extracting TensorBoard data from: {log_dir}")
    
    # Find all event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print("No TensorBoard event files found")
        return None
    
    print(f"Event files found: {len(event_files)}")
    
    # Extract data from all event files
    all_data = {}
    
    for event_file in event_files:
        print(f"Processing file: {event_file}")
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        
        # Get all scalar tags
        tags = event_acc.Tags()['scalars']
        
        for tag in tags:
            events = event_acc.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            
            if tag not in all_data:
                all_data[tag] = {'steps': steps, 'values': values}
            else:
                all_data[tag]['steps'].extend(steps)
                all_data[tag]['values'].extend(values)
    
    # Create DataFrames for each data type
    dataframes = {}
    for tag, data in all_data.items():
        # Sort by step
        steps = np.array(data['steps'])
        values = np.array(data['values'])
        sort_idx = np.argsort(steps)
        
        steps = steps[sort_idx]
        values = values[sort_idx]
        
        df = pd.DataFrame({'step': steps, 'value': values})
        dataframes[tag] = df
    
    return dataframes

def visualize_training_progress(dataframes, output_dir):
    """
    Visualizes training progress from extracted data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not dataframes:
        print("No data to visualize")
        return
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    
    if 'Loss/Generator' in dataframes:
        g_loss_df = dataframes['Loss/Generator']
        plt.plot(g_loss_df['step'], g_loss_df['value'], label='Generator Loss')
    
    if 'Loss/Discriminator' in dataframes:
        d_loss_df = dataframes['Loss/Discriminator']
        plt.plot(d_loss_df['step'], d_loss_df['value'], label='Discriminator Loss')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_losses.png'))
    plt.close()
    
    # Plot discriminator scores
    plt.figure(figsize=(12, 6))
    
    if 'Score/Real' in dataframes:
        real_df = dataframes['Score/Real']
        plt.plot(real_df['step'], real_df['value'], label='Real Scores')
    
    if 'Score/Fake' in dataframes:
        fake_df = dataframes['Score/Fake']
        plt.plot(fake_df['step'], fake_df['value'], label='Fake Scores')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Discriminator Score')
    plt.legend()
    plt.title('Discriminator Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'discriminator_scores.png'))
    plt.close()
    
    # Save data to CSV
    for tag, df in dataframes.items():
        csv_path = os.path.join(output_dir, f"{tag.replace('/', '_')}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Data saved to: {csv_path}")
    
    # Calculate basic statistics
    stats = {}
    for tag, df in dataframes.items():
        stats[tag] = {
            'mean': df['value'].mean(),
            'min': df['value'].min(),
            'max': df['value'].max(),
            'std': df['value'].std(),
            'last': df['value'].iloc[-1] if not df.empty else None
        }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.to_csv(os.path.join(output_dir, 'statistics.csv'))
    
    return stats

def create_model_from_trends(dataframes, output_dir):
    """
    Attempts to create a model with weights initialized based on observed trends
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    generator = ConditionalGenerator(
        noise_dim=100,
        signal_length=128,
        num_channels=2,
        num_mod_types=len(MOD_TYPES),
        num_sig_types=len(SIGNAL_TYPES)
    ).to(device)
    
    discriminator = ConditionalDiscriminator(
        signal_length=128,
        num_channels=2,
        num_mod_types=len(MOD_TYPES),
        num_sig_types=len(SIGNAL_TYPES)
    ).to(device)
    
    # We can't recover the exact weights, but we can save a "pseudo-checkpoint"
    # to document the training progress
    
    # Extract the latest statistics
    stats = {}
    
    if 'Loss/Generator' in dataframes:
        g_loss_df = dataframes['Loss/Generator']
        if not g_loss_df.empty:
            stats['final_g_loss'] = float(g_loss_df['value'].iloc[-1])
            stats['min_g_loss'] = float(g_loss_df['value'].min())
    
    if 'Loss/Discriminator' in dataframes:
        d_loss_df = dataframes['Loss/Discriminator']
        if not d_loss_df.empty:
            stats['final_d_loss'] = float(d_loss_df['value'].iloc[-1])
            stats['min_d_loss'] = float(d_loss_df['value'].min())
    
    # Estimate the number of epochs
    if 'Loss/Generator' in dataframes:
        g_loss_df = dataframes['Loss/Generator']
        total_steps = len(g_loss_df)
        estimated_epochs = total_steps // 13782  # Approximately 13782 steps per epoch
        stats['estimated_epochs'] = estimated_epochs
        stats['total_steps'] = total_steps
    
    # Save a pseudo-checkpoint with models and statistics
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'stats': stats,
        'timestamp': pd.Timestamp.now().isoformat(),
        'note': 'This is a pseudo-checkpoint created from TensorBoard logs. The weights do not reflect the actual training.'
    }
    
    torch.save(checkpoint, os.path.join(output_dir, 'models', 'reconstructed_model.pth'))
    print(f"Pseudo-checkpoint saved to: {os.path.join(output_dir, 'models', 'reconstructed_model.pth')}")
    
    # Save statistics to a JSON file for easy reading
    import json
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Training statistics saved to: {os.path.join(output_dir, 'training_stats.json')}")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and visualize TensorBoard data")
    parser.add_argument("--log_dir", type=str, required=True, help="TensorBoard logs directory")
    parser.add_argument("--output_dir", type=str, default="./logs_analysis", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Extract data
    dataframes = extract_tensorboard_data(args.log_dir)
    
    if dataframes:
        # Visualize training progress
        stats = visualize_training_progress(dataframes, args.output_dir)
        
        # Create model from trends
        create_model_from_trends(dataframes, args.output_dir)
    else:
        print("Could not extract data from TensorBoard logs")