# Fine-tuning F1 Race Engineer Model with QLoRA

This script fine-tunes an IBM Granite (or compatible) instruct model using QLoRA technique for efficient training with reduced memory requirements.

## Installation

Install required dependencies:

```bash
pip install -r requirements_finetune.txt
```

## Quick Start

### Basic Usage

```bash
python fine_tune_granite_qlora.py
```

### With Custom Parameters

```bash
python fine_tune_granite_qlora.py \
    --model "ibm-granite/granite-3b-instruct" \
    --dataset "f1_dataset_2024_filtered.jsonl" \
    --output "./granite_f1_finetuned" \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

## Arguments

- `--model`: Model name or path (default: `mistralai/Mistral-7B-Instruct-v0.2`)
- `--dataset`: Path to JSONL dataset (default: `f1_dataset_2024_filtered.jsonl`)
- `--output`: Output directory for fine-tuned model (default: `./granite_f1_finetuned`)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 4)
- `--learning-rate`: Learning rate (default: 2e-4)

## IBM Granite Models

If you have access to IBM Granite models, use:
- `ibm-granite/granite-3b-instruct` - 3B parameters (faster, good for testing)
- `ibm-granite/granite-7b-instruct` - 7B parameters (balanced)
- `ibm-granite/granite-8b-instruct` - 8B parameters (better performance)

## Dataset Format

The dataset should be in JSONL format with `prompt` and `completion` fields:

```json
{"prompt": "Telemetry: speed 190.4, rpm 9543.7. Advice:", "completion": "Box this lap for hards."}
```

## QLoRA Configuration

The script uses:
- **4-bit quantization** (NF4) for memory efficiency
- **LoRA rank (r)**: 16
- **LoRA alpha**: 32
- **LoRA dropout**: 0.05
- **Target modules**: All attention and MLP layers

## Memory Requirements

With QLoRA, you can fine-tune on:
- **3B model**: ~8-12 GB VRAM
- **7B model**: ~12-16 GB VRAM
- **8B model**: ~16-20 GB VRAM

## Output

The fine-tuned model will be saved to the output directory with:
- Model weights (LoRA adapters)
- Tokenizer
- Training logs
- TensorBoard logs

## Using the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3b-instruct")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3b-instruct")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./granite_f1_finetuned")

# Use the model
prompt = "Telemetry: speed 250, rpm 11000. Advice:"
inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Tips

1. **Start with 3B model** for faster iteration
2. **Monitor GPU memory** - reduce batch size if OOM errors occur
3. **Use gradient accumulation** to simulate larger batch sizes
4. **Check TensorBoard logs** for training progress: `tensorboard --logdir ./granite_f1_finetuned/logs`
