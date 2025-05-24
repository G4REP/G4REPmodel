
# G4REP: Deep Learning-Based Prediction of RNA G-Quadruplex-Binding Proteins

This repository contains the source code and scripts used in the development of **G4REP**, a deep learning framework for identifying RNA G-quadruplex-binding proteins (RG4BPs) using protein language model embeddings and LSTM-based neural networks.

## Features

- **LSTM architecture with ESM-2 embeddings**
- **High accuracy RG4BP classification**
- **Modular code for model loading, data preprocessing, and evaluation**
- **Support for full human proteome analysis**
- **Web interface available via G4REP Server**

## Files

- `Model1_ESM2_modular.py`: Modular script to load models, prepare data, and extract embeddings.
- `README.md`: Project documentation.

## Installation

```bash
pip install torch
pip install git+https://github.com/facebookresearch/esm.git
```

## Usage

```python
from Model1_ESM2_modular import main

main(
    model_name="esm1b_t33_650M_UR50S",
    fasta_file="data/input.fasta",
    output_dir="results/embeddings",
    tokens_per_batch=4096,
    seq_length=1022,
    repr_layers=[33]
)
```
