# EEG Model Training Framework

A PyTorch-based framework for training deep learning models on EEG data, with support for multiple datasets and model architectures.

## 🚀 Features

- **Multiple Model Architectures**
  - EEGConformer: Transformer-based architecture for EEG classification
  - EEGNet: Compact convolutional neural network for EEG

- **Supported Datasets**
  - BCI Competition IV 2a (motor imagery)
  - ADHD vs Control classification
  - Custom supervised EEG datasets

- **Training Infrastructure**
  - Modular training loop with built-in metrics
  - Automatic checkpointing and model saving
  - Training history visualization
  - Support for CUDA/CPU training

- **Data Processing**
  - Bandpass and notch filtering
  - Signal windowing and segmentation
  - Data normalization utilities

## 📁 Project Structure

```
eegmodel/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── data/
│   ├── raw/               # Raw EEG datasets
│   └── processed/         # Preprocessed data
├── src/
│   ├── data/              # Dataset classes and preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training loops and metrics
│   └── utils/             # Visualization and helpers
├── experiments/           # Training scripts
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for exploration
├── checkpoints/           # Saved model weights
└── requirements.txt       # Python dependencies
```

## 🔧 Installation

### Prerequisites
- Python 3.11
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JulianDPastrana/eegmodel.git
cd eegmodel
```

2. Create and activate virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 📊 Datasets

### BCI Competition IV 2a
Place the BCI IV 2a dataset files in `data/raw/BCI_IV_2a/`:
- Files: `A01T.edf`, `A01E.edf`, `A02T.edf`, etc.
- 9 subjects performing 4 motor imagery tasks

### ADHD Dataset
Place ADHD dataset files in `data/raw/EEG-DATASET-ADHD-CONTROL/`:
```
data/raw/EEG-DATASET-ADHD-CONTROL/
├── ADHD/
│   └── *.mat
└── Control/
    └── *.mat
```

## 🎯 Usage

### Quick Start

Train EEGConformer on BCI IV 2a dataset:
```bash
python experiments/train_conformer.py --dataset bci --subject 1 --epochs 50
```

Train on ADHD dataset:
```bash
python experiments/train_conformer.py --dataset adhd --epochs 50 --batch-size 32
```

### Command Line Arguments

```bash
python experiments/train_conformer.py \
    --dataset bci \              # Dataset: 'bci' or 'adhd'
    --subject 1 \                # Subject number (for BCI)
    --epochs 50 \                # Number of training epochs
    --batch-size 64 \            # Batch size
    --lr 0.001 \                 # Learning rate
    --emb-size 64 \              # Embedding dimension
    --num-heads 8 \              # Number of attention heads
    --num-layers 3 \             # Number of transformer layers
    --dropout 0.4 \              # Dropout rate
    --device cuda                # Device: 'cuda' or 'cpu'
```

### Using the API

```python
from src.models import EEGConformer
from src.data import BCIIV2aDataset
from src.training import Trainer
from src.utils import set_seed, get_device

# Set random seed
set_seed(42)

# Load dataset
dataset = BCIIV2aDataset(subject=1, split="T")

# Create model
model = EEGConformer(
    emb_size=64,
    num_channels=22,
    num_classes=4,
    num_heads=8,
    num_layers=3,
)

# Setup training
device = get_device()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

trainer = Trainer(model, optimizer, criterion, device=device)

# Train
history = trainer.fit(train_loader, val_loader, epochs=50)
```

## 🧪 Testing

Run all tests:
```bash
python -m pytest tests/
```

Run specific test:
```bash
python tests/test_models.py
python tests/test_datasets.py
python tests/test_training.py
```

## 📈 Model Architectures

### EEGConformer
Transformer-based architecture combining:
- Spatial-temporal convolution for patch embedding
- Multi-head self-attention layers
- Classification head with fully connected layers

### EEGNet
Compact CNN designed for EEG:
- Depthwise separable convolutions
- Temporal and spatial filtering
- Efficient parameter usage

## 🛠️ Development

### Adding a New Dataset

1. Create dataset class in `src/data/datasets.py`:
```python
class MyDataset(Dataset):
    def __init__(self, root_dir):
        # Load your data
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

2. Add to `src/data/__init__.py`

### Adding a New Model

1. Create model in `src/models/`:
```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Forward pass
        return output
```

2. Add to `src/models/__init__.py`

## 📝 Configuration

Edit `config/config.yaml` to customize:
- Model hyperparameters
- Training parameters
- Dataset paths
- Output directories

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Julian David Pastrana

## 🙏 Acknowledgments

- BCI Competition IV organizers
- PyTorch team
- MNE-Python developers

## 📚 References

- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.
- Song, Y., et al. (2022). EEG conformer: Convolutional transformer for EEG decoding and visualization.
