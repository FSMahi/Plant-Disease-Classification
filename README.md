# Plant Disease Classification using CNN and Transfer Learning

This repository contains implementations for training deep learning models to classify plant diseases from images. It includes a custom CNN from scratch and transfer learning using popular pretrained models: VGG19, MobileNetV2, and ResNet152.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Models](#models)  
- [Training Details](#training-details)  
- [Usage](#usage)  
- [Results](#results)  
- [Requirements](#requirements)  
- [License](#license)

---

## Project Overview

Accurate plant disease identification is essential for agriculture productivity. This project explores different deep learning architectures:

- **Custom CNN:** A convolutional neural network built from scratch for baseline performance.
- **VGG19 Transfer Learning:** Using VGG19 pretrained on ImageNet, with selective fine-tuning.
- **MobileNetV2 Transfer Learning:** Lightweight model optimized for speed and efficiency.
- **ResNet152 Transfer Learning:** Deep residual network with fine-tuning of higher layers for better accuracy.

The goal is to compare these models in terms of accuracy, training time, and computational cost on a dataset of plant leaf images.

---

## Dataset

- The dataset contains images of plant leaves classified into 15 disease categories, including healthy leaves.
- Data preprocessing includes resizing images to 224x224 pixels, normalization, and data augmentation to improve generalization.
- The dataset is split into training, validation, and test sets with stratification to maintain class balance.

---

## Models

### Custom CNN

- Multiple convolutional layers followed by max pooling.
- Fully connected layers with dropout for regularization.
- Trained from scratch on the plant disease dataset.

### VGG19 Transfer Learning

- Pretrained on ImageNet.
- The convolutional base is frozen except for the last block for fine-tuning.
- Custom dense layers added for classification.

### MobileNetV2 Transfer Learning

- Efficient architecture optimized for mobile devices.
- Only the last convolutional block is trainable.
- Custom classifier head added.

### ResNet152 Transfer Learning

- Very deep residual network.
- All layers frozen except the last convolutional block.
- Dense classifier head added on top.

---

## Training Details

- Loss function: Categorical Crossentropy.
- Optimizer: Adam with learning rate scheduling.
- Callbacks: Early stopping, ReduceLROnPlateau, ModelCheckpoint to save best weights.
- Batch size: Configurable based on hardware capabilities.
- Number of epochs: Typically 20-30 with early stopping.

---

## Results

| Model          | Test Accuracy | Test Loss | Loss Function           | Trainable Params | Notes                              |
|----------------|---------------|-----------|------------------------ |------------------|------------------------------------|
| Custom CNN     | XX%           | X.XXX     | Categorical Crossentropy| X million        | Baseline model                     |
| VGG19          | 95.75%        | 0.1365    | Categorical Crossentropy| ~22 million      | Fine-tuned last conv block         |  
| MobileNetV2    | 95.29%        | 0.1606    | Categorical Crossentropy| ~3 million       | Lightweight and faster training    |
| ResNet152      | 72.15%        | 0.8408    | Categorical Crossentropy| ~20 million      | Deep model, fine-tuning last block |

---


## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/FSMahi/plant-disease-classification.git
   cd plant-disease-classification
