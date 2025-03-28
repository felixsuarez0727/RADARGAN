# RadarGAN

RadarGAN is a Generative Adversarial Network (GAN) system designed to synthetically generate radar and communication signals. The model can produce realistic I/Q (in-phase and quadrature) signals that mimic patterns of various modulation types commonly used in radar and communication applications.

## Key Features

- I/Q signal generation for multiple modulation types
- Conditional GAN architecture for precise control of generated signal type
- Works with real (HDF5) or synthetic data
- Comprehensive evaluation of generated signal quality
- Detailed visualization of signals and training metrics
- Export of generated signals to HDF5 format

## Supported Signal Types

### Modulation Types
- AM-DSB (Amplitude Modulation - Double Sideband)
- AM-SSB (Amplitude Modulation - Single Sideband)
- ASK (Amplitude Shift Keying)
- BPSK (Binary Phase Shift Keying)
- FMCW (Frequency Modulated Continuous Wave)
- PULSED (Pulsed radar signals)

### Signal Types
- AM radio: AM broadcasting signals
- Short-range: Short-range communications
- Satellite Communication: Satellite communications
- Radar Altimeter: Radar-based altimeters
- Air-Ground-MTI: Air-Ground Moving Target Indication
- Airborne-detection: Airborne detection
- Airborne-range: Airborne range measurement
- Ground mapping: Ground terrain mapping

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.8+
- NumPy
- h5py
- matplotlib
- scikit-learn
- pandas
- seaborn
- tensorboard (optional, for visualizing training)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/RadarGAN.git
cd RadarGAN

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
RadarGAN/
├── models/
│   ├── __init__.py
│   └── gan_models.py         # Generator and Discriminator definitions
├── dataset.py                # Functions for loading real data (HDF5)
├── synthetic_dataset.py      # Synthetic data generator
├── training.py               # Training code
├── inference.py              # Signal generation and visualization
├── evaluation.py             # Metrics and evaluation
├── main.py                   # Main script
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## Usage

### Training

#### With synthetic data (no external dataset required)

```bash
python main.py train --data_file=synthetic --output_dir=./training_results --num_epochs=100 --batch_size=64
```

#### With real dataset (if available)

```bash
python main.py train --data_file=/path/to/RadComOta2.45GHz.hdf5 --output_dir=./training_results --num_epochs=100
```

#### Additional training options

```
--batch_size     # Batch size (default: 64)
--lr             # Learning rate (default: 0.0002)
--signal_length  # Signal length (default: 128)
--noise_dim      # Noise vector dimension (default: 100)
--no_cuda        # Disable GPU usage
--seed           # Seed for reproducibility (default: 42)
--beta1          # Beta1 parameter for Adam optimizer (default: 0.5)
```

### Signal Generation

#### Generate random signals

```bash
python main.py generate --checkpoint=./training_results/checkpoints/final_model.pth --num_samples=16 --output_dir=./generated_signals
```

#### Generate a specific signal type

```bash
python main.py generate --checkpoint=./training_results/checkpoints/final_model.pth --mod_type="PULSED" --sig_type="Airborne-detection" --num_samples=10 --output_dir=./pulsed_signals
```

#### Export to HDF5

```bash
python main.py generate --checkpoint=./training_results/checkpoints/final_model.pth --num_samples=100 --export_hdf5 --output_dir=./generated_signals
```

### Evaluation

```bash
python main.py evaluate --checkpoint=./training_results/checkpoints/final_model.pth --data_file=/path/to/dataset.hdf5 --output_dir=./evaluation --evaluate_by_mod
```

To also evaluate the discriminator:

```bash
python main.py evaluate --checkpoint=./training_results/checkpoints/final_model.pth --data_file=/path/to/dataset.hdf5 --evaluate_discriminator
```

## Training Monitoring

During training, visualizations and metrics are automatically generated. You can find them in:

- `./training_results/samples/`: Samples of generated signals at each epoch
- `./training_results/logs/`: Logs for TensorBoard
- `./training_results/loss_plot.png`: Plot of losses during training

To visualize metrics in real-time with TensorBoard:

```bash
tensorboard --logdir=./training_results/logs
```

## Recommended Workflow

1. **Initial Training**: Start with a brief training using synthetic data to verify everything works correctly.
   ```bash
   python main.py train --data_file=synthetic --output_dir=./test_training --num_epochs=10
   ```

2. **Full Training**: Once confirmed everything works, perform a longer training.
   ```bash
   python main.py train --data_file=synthetic --output_dir=./training_results --num_epochs=500 --batch_size=128
   ```

3. **Generation and Evaluation**: Generate signals with the trained model and evaluate their quality.
   ```bash
   python main.py generate --checkpoint=./training_results/checkpoints/final_model.pth --num_samples=100 --output_dir=./final_signals
   ```

## Troubleshooting

### Training is too slow
- Reduce batch size
- Verify you're using GPU if available
- Consider reducing signal length

### GPU memory errors
- Reduce batch size
- Disable GPU usage with `--no_cuda`

### Generated signals don't look realistic
- Increase the number of training epochs
- Adjust learning rate
- Consider modifying the model architecture in `models/gan_models.py`

## Advanced Features

### Model Customization
You can modify the generator and discriminator architecture in `models/gan_models.py` to experiment with different configurations.

### Custom Dataset Creation
If you have your own signal data, you can adapt `dataset.py` to load them in the appropriate format.

### Extension to New Modulation Types
It's possible to extend the system to support additional modulation types by adding new classes in `synthetic_dataset.py`.

## Citations and Acknowledgments

If you use this code in your research, please cite it as:

```
@software{radargan2023,
  author = {Félix Suaréz},
  title = {RadarGAN: Generative Adversarial Networks for Radar Signal Synthesis},
  year = {2023},
  url = {https://github.com/your-username/RadarGAN}
}
```

Special thanks to the RadarCommDataset repository for inspiration and data structure.
