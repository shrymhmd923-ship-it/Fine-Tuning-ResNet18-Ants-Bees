# Fine-Tuning ResNet18 — Ants vs Bees

## Overview
A computer vision project using PyTorch to fine-tune a pretrained ResNet18 model for binary image classification.

## Objective
Classify images into two categories:
- Ants
- Bees

## Methodology
- Pretrained ResNet18 with ImageNet weights
- Frozen early layers
- Fine-tuned final residual block (`layer4`)
- Replaced the original classification layer
- Cross-entropy loss
- Adam optimizer with different learning rates
- Learning-rate scheduling
- Best-checkpoint selection using validation accuracy

## Dataset
The Hymenoptera ants and bees dataset, automatically downloaded by the script when needed.

## Evaluation
The project generates:
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix
- Sample predictions with confidence scores

Validation accuracy is reported after training. Results may vary between runs.

## Technologies
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn

## How to Run
Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python fine_tuning_resnet18.py
```

The first run may download the dataset and pretrained model weights.

## Key Learning Outcomes
Transfer learning, fine-tuning, CNNs, residual networks, data augmentation, optimization, validation, and classification evaluation.
