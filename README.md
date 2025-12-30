# BPE Tokenizer Training for Python Code

This repository contains code for training a custom Byte Pair Encoding (BPE) tokenizer specifically optimized for Python code. The tokenizer is trained on a large corpus of Python functions from GitHub repositories and is designed to be more efficient for code-related tasks compared to general-purpose tokenizers like GPT-2.

## Overview

The notebook demonstrates how to:
- Load and process Python code datasets
- Train a custom BPE tokenizer using the Hugging Face `tokenizers` library
- Compare the custom tokenizer with pre-trained tokenizers (GPT-2)
- Configure tokenizer components (normalizers, pre-tokenizers, processors, decoders)

## Features

- **Code-specific tokenization**: Trained on Python code, resulting in better tokenization efficiency for code
- **Byte-level encoding**: Uses ByteLevel pre-tokenization to handle any Unicode character
- **Memory-efficient**: Processes large datasets in batches
- **Compatible with Hugging Face**: Wrapped in `PreTrainedTokenizerFast` for easy integration

## Dataset

The notebook uses the Python code dataset from [Zenodo](https://zenodo.org/records/7908468). The dataset contains:
- Python functions from GitHub repositories
- Function code, docstrings, and metadata
- Training, validation, and test splits

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install transformers datasets tokenizers
```

## Usage

1. **Download the dataset**:
   - Download `python.zip` from [Zenodo](https://zenodo.org/records/7908468)
   - Place it in the `data/` directory

2. **Run the notebook**:
   - Open `BPE_tokenization_train.ipynb` in Jupyter Notebook or Google Colab
   - Execute all cells to train the tokenizer

3. **Customize training**:
   - Adjust `vocab_size` in the `BpeTrainer` configuration (default: 25000)
   - Modify batch size in `get_training_corpus()` function (default: 1000)
   - Change special tokens as needed

## Tokenizer Configuration

- **Model**: BPE (Byte Pair Encoding)
- **Normalizer**: None (preserves Python case sensitivity)
- **Pre-tokenizer**: ByteLevel (handles Unicode characters)
- **Vocabulary Size**: 25,000 tokens
- **Special Tokens**: `<|endoftext|>`

## Results

The custom tokenizer typically produces fewer tokens for Python code compared to GPT-2's tokenizer, as it has learned code-specific subword patterns during training.

Example comparison:
- GPT-2 tokenizer: ~36 tokens for a simple Python function
- Custom tokenizer: ~27 tokens for the same function

## File Structure

```
.
├── BPE_tokenization_train.ipynb  # Main training notebook
├── README.md                      # This file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── LICENSE                       # License file
└── data/                         # Dataset directory (not included in repo)
    └── python.zip                # Dataset file (download separately)
```

## Key Components Explained

### 1. Data Loading
- Reads compressed JSONL files containing Python code
- Uses generator functions for memory efficiency
- Creates Hugging Face Dataset for easy processing

### 2. Tokenizer Training
- Initializes BPE tokenizer with ByteLevel pre-tokenization
- Trains on batches of Python code strings
- Learns frequent subword patterns in code

### 3. Tokenizer Configuration
- **Normalizer**: None (preserves code structure)
- **Pre-tokenizer**: ByteLevel (Unicode support)
- **Post-processor**: ByteLevel (handles encoding/decoding)
- **Decoder**: ByteLevel (converts tokens back to text)

## Saving and Loading the Tokenizer

To save your trained tokenizer:

```python
wrapped_tokenizer.save_pretrained("./python_code_tokenizer")
```

To load it later:

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("./python_code_tokenizer")
```

## Notes

- The notebook was originally designed for Google Colab. Adjust paths for local execution.
- Training time depends on dataset size (typically 5-15 minutes for ~400K examples).
- The tokenizer preserves Python syntax (case sensitivity, whitespace, etc.).

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [BPE Algorithm Paper](https://arxiv.org/abs/1508.07909)
- [Python Code Dataset](https://zenodo.org/records/7908468)

## Author

Created for training custom tokenizers on code datasets.

