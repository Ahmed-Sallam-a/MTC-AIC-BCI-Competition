#  MTC-AIC3 Competition Submission - EEG Classification System

This repository contains our complete submission for the **MTC-AIC3: Egypt National Artificial Intelligence Competition**, covering both **Motor Imagery (MI)** and **Steady-State Visual Evoked Potential (SSVEP)** classification tracks.

---

##  Repository Structure

```
.
├── Training/
│   ├── MI_Model.ipynb        # MI model architecture
│   └── SSVEP_Model.py                   # Final SSVEP model
├── datasets/
│   └── EEGDataset.py                  # Custom PyTorch Dataset class for both tasks
├── testing/
│   └── testing_pipeline.py        # Final inference script for both MI and SSVEP
├── models/
│   ├── best_eegnet_model.pt       # Best MI model
│   └── best_ssvep_model_ssvep.pt    # Best SSVEP model
├── sample submission/
│   └── sample_submission.csv             # Final combined predictions
├── requirements.txt
└── README.md                    
```

---

##  System Description Paper

### Overview

Our submission targets both **MI** and **SSVEP** EEG classification tasks using two specialized deep learning models with task-specific designs and tailored preprocessing.

### Model Architectures

####  Motor Imagery (MI)

* **Model**: EEGNetV4Transformer (final version in `MI_model.ipynb`)
* **Backbone**: EEGNet-style CNN with Squeeze-and-Excitation blocks
* **Enhancement**: Lightweight Transformer encoder for temporal attention
* **Dropout**: 0.5 to prevent overfitting
* **Accuracy**: \~67% Train / \~67% Validation

#### SSVEP

* **Model**: EEGNetPro (final version in `SSVEP_Model.ipynb`)
* **Architecture**: EEGNet variant optimized for SSVEP with spatial filters and SE attention
* **Signal Length**: 1750 samples per trial
* **Classes**: 4 visual stimuli frequencies
* **Accuracy**: Comparable performance with better generalization than transformer-based models

---

###  Preprocessing & Technical Details

* **Filtering**: Bandpass filter applied to raw EEG
* **Normalization**: Trial-wise z-score normalization, with optional global standardization
* **Augmentation**: Gaussian noise (train only)
* **Frameworks**: PyTorch, scikit-learn, pandas
* **Devices**: Trained on GPU via Kaggle Notebooks

---

### Challenges & Solutions

| Challenge                             | Solution                                                              |
| ------------------------------------- | --------------------------------------------------------------------- |
| Transformer overfitting on small data | Reduced layers, added SE blocks, and applied dropout                  |
| Small dataset                         | Heavy regularization, Mixup (optional), trial normalization           |
| Label decoding for test               | Used shared `LabelEncoder` fit from training folds                    |
| Inference consistency                 | Unified prediction pipeline for both MI and SSVEP with dynamic loader |

---

## Inference Pipeline

The script `inference/generate_submission.py` loads both models and generates a single CSV with:

```
id,label
MI_001,Left
MI_002,Right
SSVEP_001,13Hz
...
```

* Uses same dataset logic (`EEGDataset`) as training
* Ensures reproducibility by using `torch.load()` for model weights
* Final predictions are saved in `sample_submission.csv`

---

## Training Pipeline

Both training scripts (`MI_Model.ipynb` and `train_ssvep.ipynb`) include:

* Dataset split
* Dataloader logic with transforms
* Training loop with early stopping and validation
* Model saving with best weights
* Optional logging of accuracy/loss per epoch

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Model Architectures

### Motor Imagery (MI): EEGNetV4Transformer

Our MI classification model is based on an enhanced EEGNet architecture augmented with attention and temporal modeling using Transformers. The architecture consists of the following components:

#### ➤ 1. First Temporal Convolution

* `Conv2d(1, 16, (1, 32), padding=(0, 16))`
  Captures temporal patterns with a wide receptive field.
* `BatchNorm2d(16)`

#### ➤ 2. Depthwise Spatial Convolution

* `Conv2d(16, 32, (8, 1), groups=16)`
  Learns spatial filters for each temporal feature map across all 8 EEG channels.
* `BatchNorm2d(32)`
* `ELU`
* `AvgPool2d((1, 4))`
* `Dropout(p=0.5)`

#### ➤ 3. Separable Convolution

* `Conv2d(32, 32, (1, 16), padding=(0, 8))`
  Extracts abstract temporal features with fewer parameters.
* `BatchNorm2d(32)`
* `ELU`
* `AvgPool2d((1, 8))`
* `Dropout(p=0.5)`

#### ➤ 4. Squeeze-and-Excitation Block

A channel-wise attention mechanism that recalibrates feature maps by modeling inter-channel dependencies.

* Adaptive Average Pool → 2 Fully Connected layers → Sigmoid gate
* Output: channel-wise attention weights applied to each feature map

#### ➤ 5. Transformer Encoder (1 layer)

* Input shape: `[B, T', C]` where `T'` is the temporal resolution after convolutions
* `TransformerEncoderLayer(d_model=32, nhead=4)`
* Adds positional encoding
* Models long-range temporal dependencies across feature maps

#### ➤ 6. Global Average Pooling + Classifier

* `x = x.mean(dim=1)` (averages across temporal sequence)
* `Linear(32 → num_classes)`

#### Output:

* **2-class softmax**: Corresponds to the MI tasks (e.g., Left vs Right)

---

### Steady-State Visual Evoked Potential (SSVEP): EEGNetPro

For SSVEP classification, we used a compact and highly regularized variant of EEGNet that performed best with minimal overfitting. It consists of:

#### ➤ 1. Temporal Convolution

* `Conv2d(1, 16, (1, 64), padding=(0, 32))`
* `BatchNorm2d(16)`

#### ➤ 2. Spatial Depthwise Convolution

* `Conv2d(16, 32, (8, 1), groups=16)`
* `BatchNorm2d(32)`
* `ELU`
* `AvgPool2d((1, 4))`
* `Dropout(p=0.5)`

#### ➤ 3. Separable Convolution

* `Conv2d(32, 32, (1, 16), padding=(0, 8))`
* `BatchNorm2d(32)`
* `ELU`
* `AvgPool2d((1, 4))`
* `Dropout(p=0.5)`

#### ➤ 4. Squeeze-and-Excitation Attention

Improves spatial-channel interactions critical in SSVEP tasks where frequencies are spatially encoded.

#### ➤ 5. Flatten + Classifier

* Global Average Pool across time
* `Linear(32 → 4)` for 4 visual stimulus classes

#### Output:

* **4-class softmax**: Corresponds to SSVEP frequency classes (e.g., 10Hz, 12Hz, 13Hz, 15Hz)

---

Both models share a consistent preprocessing pipeline, enabling unified code for inference and training while optimizing performance per task.

---


