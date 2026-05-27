# Multilingual Neural Machine Translation Model

A production-ready neural machine translation (NMT) system built with [OpenNMT](https://opennmt.net/) that translates from **English, French, and German** to **Vietnamese**. The model is optimized for training on Google's A100 GPU via Google Colab.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Preparation Pipeline](#data-preparation-pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)

## Overview

This project implements a multilingual-to-one (M2O) machine translation model using OpenNMT-py. The system handles translation from three European languages (English, French, German) to Vietnamese, with careful attention to:

- **Data cleaning and filtering** - Removing noise, duplicates, and malformed segments
- **Subword tokenization** - Using SentencePiece for handling out-of-vocabulary (OOV) words
- **Optimized training** - Configured for efficient training on high-end GPUs
- **Model evaluation** - BLEU score computation for quality assessment

## Project Structure

```
multilingual-model/
├── README.md                              # This file
├── compute-bleu.py                        # BLEU score evaluation script
├── model_released.pt                      # Pre-trained model weights
├── ONMT-Training.ipynb                    # Google Colab notebook for training
├── ONMT-Data-Processing.ipynb             # Google Colab notebook for data prep
├── MT-Preparation/                        # Data preparation pipeline
│   ├── requirements.txt                   # Python dependencies
│   ├── filtering/
│   │   ├── filter.py                      # Dataset cleaning script
│   │   └── semantic_filter.py             # Advanced semantic filtering
│   ├── subwording/
│   │   ├── 1-train_bpe.py                 # Train BPE subword model
│   │   ├── 1-train_unigram.py             # Train unigram (SentencePiece) model
│   │   ├── 1-train_unigram_joint.py       # Joint unigram training
│   │   ├── 2-subword.py                   # Apply subword tokenization
│   │   ├── 3-desubword.py                 # Reverse subword tokenization
│   │   └── spm_to_vocab.py                # Convert SentencePiece to OpenNMT vocab
│   └── train_dev_split/
│       ├── train_dev_split.py             # Split data into train/dev sets
│       └── train_dev_test_split.py        # Split data into train/dev/test sets
└── MT-Preparation/README.md               # Detailed pipeline documentation
```

## Getting Started

### Prerequisites

- Python 3.7+
- pip or conda package manager
- GPU access (CUDA-compatible for optimal performance)
- ~50GB+ disk space (depending on corpus size)

### Installation

1. **Install MT-Preparation dependencies:**

```bash
cd MT-Preparation
pip install -r requirements.txt
```

2. **Install OpenNMT-py (if training locally):**

```bash
pip install OpenNMT-py
```

## Data Preparation Pipeline

The MT-Preparation directory contains a complete pipeline for preparing your multilingual translation dataset:

### Step 1: Data Filtering

Clean your parallel corpora using the filtering script:

```bash
python filter.py <source_file> <target_file> <source_lang> <target_lang>
```

**Processing includes:**
- Removing empty and duplicate segments
- Filtering source-copied (low-quality) translations
- Removing HTML and malformed text
- Length filtering (with 200% ratio threshold)
- Optional lowercasing
- Dataset shuffling

### Step 2: Subword Tokenization

Train SentencePiece models to handle subword units:

```bash
cd subwording
python 1-train_unigram.py <train_source_file> <train_target_file>
```

Then apply the models to tokenize your data:

```bash
python 2-subword.py <source_model> <target_model> <source_file> <target_file>
```

**Notes:** SentencePiece handles rare words and multilingual vocabulary efficiently, reducing OOV issues.

### Step 3: Train/Dev/Test Split

Split your processed data:

```bash
python train_dev_split.py <num_dev_segments> <source_file> <target_file>
```

For a three-way split (train/dev/test):

```bash
python train_dev_test_split.py <num_dev_segments> <num_test_segments> <source_file> <target_file>
```

## Training

The project includes two Google Colab notebooks for easy training:

1. **ONMT-Data-Processing.ipynb** - Runs the complete data preparation pipeline
2. **ONMT-Training.ipynb** - Trains the neural machine translation model

### To train locally (advanced):

```bash
# Build vocabulary
onmt-build-vocab -config config.yaml

# Train model
onmt-main train -config config.yaml
```

## Evaluation

Compute BLEU scores on test data:

```bash
python compute-bleu.py <reference_file> <hypothesis_file>
```

## Usage

### Translating with the pre-trained model

```python
# Example using OpenNMT-py
from onmt.translate import Translator

translator = Translator.from_model('model_released.pt')
predictions = translator.translate(['<s> source text here </s>'])
```

### Desubwording predictions

After translation, reverse the SentencePiece tokenization:

```bash
python MT-Preparation/subwording/3-desubword.py <target_model> <translations_file>
```

## Model Architecture

- **Framework:** OpenNMT-py
- **Architecture:** Transformer encoder-decoder
- **Tokenization:** SentencePiece (unigram)
- **Training Hardware:** Google A100 GPU (recommended)
- **Supported Language Pairs:** EN→VI, FR→VI, DE→VI

## Resources

- [OpenNMT Documentation](https://opennmt.net/)
- [SentencePiece GitHub](https://github.com/google/sentencepiece)
- [OpenNMT-py GitHub](https://github.com/OpenNMT/OpenNMT-py)

## License

Project structure and scripts created for machine translation research.