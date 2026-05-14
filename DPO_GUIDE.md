# Direct Preference Optimization (DPO) Training Guide

This guide explains how to use the DPO training implementation for the NeuroQuantum model.

## Overview

**DPO (Direct Preference Optimization)** is a training technique that directly optimizes models to prefer chosen responses over rejected responses, without requiring a separate reward model. This is particularly useful for aligning language models with human preferences.

### Key Benefits
- **Simpler than RLHF**: No need for a separate reward model
- **More sample-efficient**: Uses preference pairs directly
- **Stable training**: Direct loss optimization without RL instability

## DPO Loss Function

The DPO loss is:

```
DPO_loss = -log(sigmoid(β * (log p_θ(y_c|x) - log p_θ(y_r|x))))
```

Where:
- `p_θ` = model's probability distribution
- `y_c` = chosen (preferred) response
- `y_r` = rejected response
- `β` = temperature parameter controlling preference strength
- `x` = input prompt

## Training Scripts

### 1. Pure DPO Training (`train_dpo.py`)

Train the model using only preference pairs.

```bash
python train_dpo.py
```

**Configuration:**
- `EPOCHS`: 5 (number of training epochs)
- `LR`: 1e-5 (learning rate)
- `BATCH_SIZE`: 2 (preference pairs per batch)
- `DPO_BETA`: 0.5 (preference temperature)

**Data Sources:**
- argilla/ultrafeedback-binarized (HuggingFace)
- Hand-crafted preference examples

### 2. Combined Training (`train_combined_dpo.py`)

Two-phase training: first QA, then DPO.

```bash
python train_combined_dpo.py
```

**Phase 1 - QA Training (2 epochs):**
- Trains on standard QA pairs
- Learns basic question-answering patterns
- Uses datasets: Alpaca, FreedomIntelligence

**Phase 2 - DPO Fine-tuning (3 epochs):**
- Aligns model with preference data
- Refines response quality based on preferences

This approach combines the benefits of both training methods:
1. Strong QA foundation from supervised learning
2. Preference alignment from DPO

## Data Format

### Preference Pairs

Each sample requires:
```python
{
    "prompt": "What is machine learning?",
    "chosen": "Machine learning is a subfield of AI...",
    "rejected": "It's about computers."
}
```

### Supported Datasets

- **HuggingFace Datasets**: Auto-detection of `prompt`, `chosen`, `rejected` fields
- **Custom Datasets**: Load from dictionaries or JSON files
- **Hand-crafted Examples**: Built-in preference examples for testing

## DPO Metrics

During training, the model tracks:

- **Loss**: DPO loss value (lower is better)
- **Accuracy**: Percentage of samples where log_p(chosen) > log_p(rejected)
- **Avg Log Ratio**: Average difference in log probabilities

## Hyperparameter Guide

### DPO Beta (`β`)
- **Lower (0.1-0.3)**: Softer preference learning, more exploration
- **Medium (0.5-1.0)**: Balanced preference alignment
- **Higher (1.0+)**: Stronger preference enforcement

### Learning Rate
- **DPO LR**: 1e-5 to 5e-5 (usually lower than QA training)
- **Warmup Steps**: 20 (gradual learning rate increase)

### Batch Size
- Reduce if GPU memory is limited
- Larger batches provide better loss estimates

## Implementation Details

### DPO Utilities (`dpo_utils.py`)

Key functions:
- `load_preference_data_from_hf()`: Load from HuggingFace
- `tokenize_preference_pair()`: Encode preference pairs
- `compute_dpo_loss()`: Calculate DPO loss
- `compute_dpo_metrics()`: Track training metrics

### Preference Handling

1. **Tokenization**: Both chosen and rejected responses are tokenized
2. **Padding**: Aligned to same length for batch processing
3. **Log Probability Computation**: Per-token cross-entropy converted to log-probs
4. **Loss Computation**: Compares chosen vs rejected log-prob sums

## Example Usage

### Custom Preference Data

```python
from dpo_utils import load_preference_data_from_dict

preferences = [
    {
        "prompt": "何は機械学習ですか?",
        "chosen": "機械学習は...",
        "rejected": "機械学習は古い技術です"
    }
]

data = load_preference_data_from_dict(preferences)
```

### Training with Different Beta

Edit `train_dpo.py`:
```python
DPO_BETA = 1.0  # Stronger preference enforcement
```

## Monitoring Training

The training logs include:
- Loss per epoch
- Preference accuracy
- Timestamp
- Average log probability ratio

Check saved checkpoint for training history:
```python
import torch
ckpt = torch.load("neuroq_checkpoint.pt")
logs = ckpt["training_log"]
# Filter DPO-phase logs
dpo_logs = [l for l in logs if l.get("phase") == "DPO"]
```

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE`
- Reduce `max_seq_len`
- Use gradient accumulation with smaller steps

### Poor Performance
- Check data quality (preference pairs should be clearly distinct)
- Increase `DPO_BETA` if model doesn't distinguish responses
- Ensure preference data is diverse

### Loss Not Decreasing
- Try lower learning rate (1e-5)
- Increase number of epochs
- Verify preference data quality

## Next Steps

1. **Collect Preference Data**: Gather human feedback on model outputs
2. **Prepare Datasets**: Convert to required format
3. **Run Training**: Start with `train_combined_dpo.py`
4. **Evaluate**: Test model outputs for preference alignment
5. **Iterate**: Collect more feedback and refine

## References

- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
- Implementation based on preference optimization principles from RLHF research

## File Structure

```
Qubit/
├── dpo_utils.py              # DPO utilities and loss computation
├── train_dpo.py              # Pure DPO training
├── train_combined_dpo.py      # Combined QA + DPO training
├── neuroq_checkpoint.pt       # Model checkpoint
└── DPO_GUIDE.md             # This file
```
