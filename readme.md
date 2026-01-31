# MiniGPT-v1

A minimal implementation of a GPT-style Transformer model using PyTorch. This project demonstrates the core components of the Transformer architecture, including Multi-Head Self-Attention, Feed-Forward networks, and residual connections, applied to a small text corpus.

## Project Structure

* `transformer_blocks.py`: Contains the modular architectural components of the Transformer.
    * `SelfAttentionHead`: Implements a single head of scaled dot-product attention with causal masking.
    * `MultiHeadAttention`: Orchestrates multiple attention heads in parallel.
    * `FeedForward`: A simple linear-layer network with ReLU activation.
    * `Block`: A complete Transformer block combining layer normalization, multi-head attention, and feed-forward networks with residual connections.
* `v1.py`: The main entry point for training and generation.
    * Handles data preprocessing and vocabulary creation from a sample corpus.
    * Defines the `TinyGPT` model class.
    * Implements the training loop and text generation logic.
* `a.py`: A utility script to verify the PyTorch installation and version.

## Model Architecture

The `TinyGPT` model consists of:
1.  **Token Embeddings**: Maps input word indices to dense vectors.
2.  **Position Embeddings**: Provides the model with information about the relative or absolute position of words in a sequence.
3.  **Transformer Blocks**: A stack of sequential blocks (Multi-Head Attention + Feed Forward).
4.  **Language Modeling Head**: A final linear layer that maps the transformer output back to the vocabulary size to predict the next word.

## Getting Started

### Prerequisites
* Python 3.x
* PyTorch

### Training the Model
To train the model on the provided sample corpus, run:
```bash
python v1.py