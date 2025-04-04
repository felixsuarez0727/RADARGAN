import argparse
import os
import sys
import torch

def main():
    """
    Main function to handle all subcommands
    """
    parser = argparse.ArgumentParser(
        description="RadarGAN: GAN for generating radar and communication signals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Command: train
    train_parser = subparsers.add_parser("train", help="Train a new GAN")
    train_parser.add_argument("--data_file", type=str, required=True, help="Path to HDF5 dataset file or 'synthetic'")
    train_parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    train_parser.add_argument("--signal_length", type=int, default=128, help="Signal length")
    train_parser.add_argument("--noise_dim", type=int, default=100, help="Noise vector dimension")
    train_parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    train_parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    train_parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    train_parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    train_parser.add_argument("--resume_from_checkpoint", type=str, help="Path to a checkpoint to resume training from")

    # Command: generate
    gen_parser = subparsers.add_parser("generate", help="Generate signals using a trained model")
    gen_parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    gen_parser.add_argument("--num_samples", type=int, default=10, help="Number of signals to generate")
    gen_parser.add_argument("--output_dir", type=str, default="./generated", help="Directory to save generated signals")
    gen_parser.add_argument("--mod_type", type=str, help="Specific modulation type to generate")
    gen_parser.add_argument("--sig_type", type=str, help="Specific signal type to generate")
    gen_parser.add_argument("--export_hdf5", action="store_true", help="Export signals to HDF5")
    gen_parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    eval_parser.add_argument("--data_file", type=str, required=True, help="Path to HDF5 dataset file for evaluation")
    eval_parser.add_argument("--output_dir", type=str, default="./evaluation", help="Directory to save evaluation")
    eval_parser.add_argument("--evaluate_by_mod", action="store_true", help="Evaluate by modulation type")
    eval_parser.add_argument("--evaluate_discriminator", action="store_true", help="Evaluate discriminator")
    eval_parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Device selection
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute the corresponding command
    if args.command == "train":
        from training import train_gan
        print(f"== Starting RadarGAN training on {device} ==")
        
        # Add device to args
        args.device = device
        
        # Check if synthetic data is requested
        if args.data_file.lower() == "synthetic":
            args.use_synthetic = True
            print("Using synthetic data for training")
        else:
            args.use_synthetic = False
        
        train_gan(args)
    
    elif args.command == "generate":
        from inference import load_generator, generate_signals, visualize_generated_signals, export_signals_to_hdf5
        print("== Generating signals with RadarGAN ==")
        
        # Configure device
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load generator
        generator = load_generator(args.checkpoint, device)
        
        # Configure specific types if requested
        specific_config = None
        if args.mod_type is not None or args.sig_type is not None:
            from dataset import MOD_TYPES, SIGNAL_TYPES
            specific_config = {}
            if args.mod_type is not None:
                if args.mod_type in MOD_TYPES:
                    specific_config['mod_type'] = MOD_TYPES[args.mod_type]
                else:
                    print(f"Unknown modulation type: {args.mod_type}")
                    print(f"Available types: {list(MOD_TYPES.keys())}")
                    sys.exit(1)
            if args.sig_type is not None:
                if args.sig_type in SIGNAL_TYPES:
                    specific_config['sig_type'] = SIGNAL_TYPES[args.sig_type]
                else:
                    print(f"Unknown signal type: {args.sig_type}")
                    print(f"Available types: {list(SIGNAL_TYPES.keys())}")
                    sys.exit(1)
        
        # Generate signals
        generated_signals, mod_types, sig_types = generate_signals(
            generator, 
            num_samples=args.num_samples,
            device=device,
            specific_config=specific_config
        )
        
        # Visualize signals
        visualize_generated_signals(
            generated_signals,
            mod_types,
            sig_types,
            output_path=os.path.join(args.output_dir, 'generated_signals.png')
        )
        
        # Export to HDF5 if requested
        if args.export_hdf5:
            export_signals_to_hdf5(
                generated_signals,
                mod_types,
                sig_types,
                output_file=os.path.join(args.output_dir, 'generated_signals.hdf5')
            )
    
    elif args.command == "evaluate":
        from evaluation import evaluate_by_modulation, evaluate_adversarial_discriminator
        from inference import load_generator
        from dataset import get_dataloader, MOD_TYPES, SIGNAL_TYPES
        from models import ConditionalDiscriminator
        
        print("== Evaluating RadarGAN ==")
        
        # Configure device
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load generator
        generator = load_generator(args.checkpoint, device)
        
        # Load real data
        dataloader = get_dataloader(
            hdf5_file=args.data_file,
            batch_size=64,
            signal_length=128,
            normalize=True,
            shuffle=True,
            num_workers=4
        )
        
        # Evaluate by modulation type if requested
        if args.evaluate_by_mod:
            metrics_df = evaluate_by_modulation(
                generator,
                dataloader,
                device,
                os.path.join(args.output_dir, 'by_modulation')
            )
            print("\nMetrics by modulation type:")
            print(metrics_df)
        
        # Evaluate discriminator if requested
        if args.evaluate_discriminator:
            # Load discriminator from the same checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=device)
            discriminator = ConditionalDiscriminator(
                signal_length=128,
                num_channels=2,
                num_mod_types=len(MOD_TYPES),
                num_sig_types=len(SIGNAL_TYPES)
            ).to(device)
            
            # Check if checkpoint contains the discriminator
            if 'discriminator_state_dict' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                print("Discriminator loaded from checkpoint")
                
                metrics = evaluate_adversarial_discriminator(
                    generator,
                    discriminator,
                    dataloader,
                    device,
                    os.path.join(args.output_dir, 'discriminator')
                )
                print("\nDiscriminator metrics:")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")
            else:
                print("Discriminator state dict not found in checkpoint, skipping evaluation")

if __name__ == "__main__":
    main()