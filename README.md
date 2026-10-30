# SmolLM2

A lightweight project for working with the SmolLM2 data corpus and model setup in a local Python environment.

## Overview

This repository includes:

- a local dataset directory with train/validation/test/evaluation JSONL files
- a notebook for checking data availability and environment setup
- a Python requirements file for installing the required dependencies
- a standard Git ignore configuration for Python projects

## Project structure

```text
SmolLM2/
├── dummy_corpus/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── general_eval.jsonl
├── requirements.txt
├── .gitignore
├── smollm2.ipynb
├── README.md
└── llama.cpp/
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook and run the cells to verify that the dataset files are present.

## Data

The project expects the following files under `dummy_corpus/`:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `general_eval.jsonl`

A notebook check validates that all files exist before continuing.

## Model architecture

The PyTorch model implemented in the notebook follows a decoder-only causal language model architecture inspired by LLaMA. It is configured to match the Hugging Face SmolLM2-135M checkpoint:

- vocabulary size: 49,152
- hidden size: 576
- intermediate size: 1,536
- layers: 30
- attention heads: 9
- KV heads: 3
- max context length: 8,192
- positional encoding: Rotary Positional Embeddings (RoPE)
- normalization: RMSNorm
- attention pattern: grouped-query attention (GQA)
- activation: GELU / MLP expansion with residual connections
- output head: causal LM head with tied embedding weights

In other words, the implementation stacks 30 Transformer decoder blocks, each performing pre-norm residual attention and an MLP block, and the network predicts the next token autoregressively over a sequence.

## Notes

This repository is intended for local experimentation and model-development workflows using the SmolLM2 dataset setup.
