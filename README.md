
# U-Net for Medical Image Segmentation

A PyTorch implementation of the **U-Net convolutional neural network** for binary medical image segmentation. This project explores pixel-level segmentation using an encoder–decoder architecture with skip connections, trained and evaluated using a notebook-based workflow.

## Overview

U-Net is a widely used architecture for biomedical image segmentation. It combines:

- **Encoder:** Extracts increasingly abstract features while reducing spatial dimensions.
- **Bottleneck:** Captures high-level contextual information.
- **Decoder:** Recovers spatial resolution to produce a segmentation mask.
- **Skip connections:** Concatenate encoder feature maps with corresponding decoder features to preserve spatial details and improve localization.

This implementation uses convolutional blocks containing two convolution layers, each followed by batch normalization and ReLU activation.

## Architecture

The network consists of:

- Four encoder stages with feature channels of 64, 128, 256, and 512
- A bottleneck with 1024 channels
- Four decoder stages using transposed convolutions for upsampling
- Skip connections between encoder and decoder stages
- A final 1×1 convolution to produce a single-channel segmentation map
- Sigmoid activation to generate pixel-wise probabilities

The model accepts RGB images and predicts a binary segmentation mask.

## Dataset

The notebook is configured to load a histopathology segmentation dataset from a Kaggle input directory.

**Dataset path configured in the notebook:**

```text
/kaggle/input/datasets/kartikmaity/tnbc-monuseg-segmentation/
TNBC/TNBC/TNBC_without_split/
├── Images/
└── Masks/
```

The dataset loader reads images as RGB and masks as grayscale. Masks are converted to binary values using a threshold.

## Data Preprocessing and Augmentation

The notebook includes:

- Image and mask loading
- Binary mask conversion
- Resizing to 512 × 512 for the augmentation pipeline
- Horizontal and vertical flipping
- Random 90-degree rotations
- Small shifts, scaling, and rotations
- Tensor conversion for PyTorch training

The dataset is randomly divided into:

| Split | Proportion |
|---|---:|
| Training | 70% |
| Validation | 20% |
| Testing | 10% |

The split is randomized using a fixed seed for reproducibility.

## Training

The model is implemented in **PyTorch** and trained using a custom training loop.

Training components include:

- **Loss function:** Combined Binary Cross-Entropy and Dice loss
- **Optimizer:** Defined in the notebook training pipeline
- **Epochs:** Up to 100
- **Best-model checkpointing:** Saves the model with the lowest validation loss
- **Device support:** CUDA GPU when available, otherwise CPU

Training and validation loss, Dice coefficient, and IoU are tracked across epochs. The notebook also plots training and validation curves.

## Evaluation Metrics

The model is evaluated using:

- **Dice coefficient:** Measures overlap between predicted and ground-truth masks.
- **Intersection over Union (IoU):** Measures the ratio of intersection to union between the predicted and ground-truth regions.

The test evaluation uses a probability threshold of **0.5** to convert predictions into binary masks.

## Repository Contents

```text
U-Net/
├── README.md
└── unet-implementation.ipynb
```

The notebook contains the dataset loading, preprocessing, model definition, training, metric calculation, checkpoint loading, and test evaluation workflow.

## How to Run

1. Open the notebook in Kaggle or a compatible Jupyter environment.
2. Attach the required dataset.
3. Verify and update the dataset path.
4. Ensure the required Python packages are installed.
5. Run the notebook cells in order.
6. Review the training curves and test-set Dice and IoU results.

## Requirements

The notebook uses:

- Python
- PyTorch
- NumPy
- Matplotlib
- Pillow
- Albumentations
- OpenCV
- tqdm
- torchinfo

## Purpose

This U-Net implementation serves as a baseline for medical image segmentation and provides a reference point for comparison with more advanced models in this repository, including U-Net++, attention-based U-Net variants, TransUNet, self-attention models, and dual-encoder architectures.

## Notes

- The notebook currently uses a binary segmentation setup with one output channel.
- Dataset availability and usage are subject to the dataset provider’s terms.
- The notebook’s reported metrics should be added here after running the final experiment.

## Author

**Soumyadeepa Dutta**

Repository: [Medical Image Segmentation](../)
