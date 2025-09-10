# CIFAR-10 Classification with Modified ResNet

A PyTorch implementation of a modified ResNet architecture for classifying a subset of CIFAR-10 images with custom activation function replacement and comprehensive evaluation.

## Project Overview

This project demonstrates:
- Modification of ResNet architecture by replacing activation functions
- Filtering CIFAR-10 dataset to specific classes
- Training and evaluation of a deep learning model
- Comprehensive visualization and analysis of results

## Features

- **Model Architecture**: Modified ResNet18 with custom activation function replacement
- **Data Processing**: Filtering CIFAR-10 to selected classes (cat, dog, airplane)
- **Activation Replacement**: Systematic replacement of ReLU activations with SiLU from the 7th occurrence onward
- **Training Pipeline**: Complete training loop with validation and learning rate scheduling
- **Visualization**: Sample images, training curves, and prediction analysis
- **Reproducibility**: Fixed random seeds and deterministic behavior

## Requirements

- Python 3.9+
- PyTorch 2.x
- torchvision 0.15+
- Other dependencies: numpy, matplotlib, scikit-learn, seaborn, tqdm, Pillow

## Installation

1. **Create virtual environment**:
   ```bash
   python -m venv cifar10_env
   source cifar10_env/bin/activate  # On Windows: cifar10_env\Scripts\activate


2. **Install dependencies**:
     ```bash
   pip install torch torchvision numpy matplotlib scikit-learn seaborn tqdm Pillow jupyter ipykernel

**Project Structure**

  ```bash
cifar10-resnet-project/
├── cifar10_resnet_modification.ipynb  # Main notebook
├── requirements.txt                   # Dependencies
├── data/                             # CIFAR-10 dataset (auto-downloaded)
└── README.md
```
**Usage**
**1. Run the Jupyter notebook**:
  ```bash
jupyter notebook cifar10_resnet_modification.ipynb
```
Overview
This notebook demonstrates modifying a pretrained ResNet18 model by replacing ReLU activations with SiLU, then training it on a 3-class subset of CIFAR-10.

Implementation Details
Model Architecture
Base Model: ResNet18 with pretrained weights

Activation Replacement: Replaces ReLU with SiLU from the 7th occurrence (0-based index 6)

Classification Head: Modified for 3-class output (original: 1000 classes)

Data Pipeline
Dataset: CIFAR-10 filtered to 3 classes: ['cat', 'dog', 'airplane']

Transforms:

Training: Random crop, horizontal flip, normalization

Validation: Center crop, normalization

Label Mapping: Original labels mapped to contiguous IDs

Training Configuration
Loss Function: CrossEntropyLoss

Optimizer: Adam with learning rate 0.001

Scheduler: StepLR (gamma=0.1, step_size=7)

Batch Size: 64

Epochs: 10

Expected Results
Model Architecture
text
Total nn.ReLU count before replacement: 9
Replaced activations at 0-based indices: [6, 7, 8]
Replaced activations at 1-based indices: [7, 8, 9]
ReLU remaining: 6
SiLU inserted: 3
Dataset Statistics
text
Selected classes: ['cat', 'dog', 'airplane']
Training set counts:
  Class 0 (cat): 5000 samples
  Class 1 (dog): 5000 samples
  Class 2 (airplane): 5000 samples
Test set counts:
  Class 0 (cat): 1000 samples
  Class 1 (dog): 1000 samples
  Class 2 (airplane): 1000 samples
Performance
Training time: ~8-10 minutes on CPU

Final validation accuracy: ~80-85%

Best validation accuracy: Typically achieved in later epochs

Output Visualizations
Sample Augmented Images: Display of training images with augmentations

Training Curves: Loss and accuracy plots over epochs

Confusion Matrix: Performance per class

Test Examples: Predictions with confidence scores

Misclassified Examples: Analysis of model errors

Customization Options
Changing Selected Classes
python
selected_classes = ['cat', 'dog', 'bird']  # Example change
Using Different Activation Functions
python
# Options: nn.SiLU(), nn.GELU(), nn.LeakyReLU(0.01)
model = count_and_replace_relu(model, replace_from_idx=6, new_activation_class=nn.GELU)
Adjusting Training Parameters
python
num_epochs = 15
optimizer = optim.Adam(model.parameters(), lr=0.0005)
batch_size = 128
Troubleshooting
Blurry images: Normal for CIFAR-10's 32x32 resolution

Download errors: Check internet connection for dataset download

Memory issues: Reduce batch size if needed

Reproducibility
Fixed random seed (42) for all libraries

CUDA deterministic operations enabled

Environment variables set for consistent behavior

Total runtime: ≤ 20 minutes on CPU

License
This project is for educational purposes. Feel free to modify and use as needed.

Acknowledgments
CIFAR-10 dataset by Alex Krizhevsky

PyTorch and torchvision teams

ResNet architecture by He et al.





