# Fine-tuning F1 Race Engineer Model with QLoRA

This project contains two F1 race engineer training datasets:

1. **Live Race Engineer** - Real-time radio responses during the race (using team radio + telemetry)
2. **Post-Race Analysis** - Detailed technical debriefs after the race (using telemetry + race events)

## Dataset Generation

### 1. Live Race Engineer Dataset 

Uses OpenF1 API + Whisper to pair team radio transcripts with telemetry context:

```bash
python build_f1_race_engineer_dataset.py
```

**Output:** `f1_dataset_combined_filtered.jsonl`

This dataset teaches the model to respond like a race engineer during the race - short, actionable radio communications based on live telemetry.

### 2. Post-Race Analysis Dataset

Uses FastF1 (free F1 telemetry library) and Gemini API to create professional engineering analyses:

```bash
python build_post_race_dataset_fastf1.py
```

**What it does:**
- Fetches telemetry data for each driver in each race (2023/2024/2025 seasons)
- Sends telemetry + race events to Gemini for analysis
- Generates training examples: telemetry input → engineering debrief output
- Handles data quality issues intelligently (missing laps, incomplete stints)
- Network error resilience (skips failed drivers, resumes from last successful)

**Output:** 
- `post_race_training_data_2023.jsonl` (~440 examples)
- `post_race_training_data_2024.jsonl` (~440 examples)
- `post_race_training_data_2025.jsonl` (~440 examples)
- **Total:** ~1,320 training examples

**Hallucination Prevention:**
Critical prompt engineering to prevent Gemini from citing race events the fine-tuned model won't have access to:
- Race events marked "FOR CONTEXT ONLY" in Gemini prompt
- Requires inferential language ("suggests", "likely", "may indicate") instead of definitive statements
- Prohibits quoting race control messages, penalties, or specific incidents by name
- All conclusions must be derivable from telemetry alone

**Example:**
- ALLOWED: "Lap 28 spike (114s) suggests Safety Car deployment"
- FORBIDDEN: "The penalty on Lap 14 for track limits..."

**Prompt Design:**
- Gemini receives full telemetry + race events (more context than the fine-tuned model will get at inference)
- Prompt instructs natural handling of data limitations (mention missing laps, incomplete stints matter-of-factly within relevant sections)
- Teaches the model to work with imperfect data like a real race engineer would
- Generates 9-section structured analysis: Performance, Pace, Tire Management, Strategy, Strengths, Weaknesses, Comparison, Car Data, Recommendations


This dataset teaches the model to provide comprehensive post-race technical debriefs with detailed quantitative analysis.

### Dataset Formats

**Live Race Dataset:**
```json
{"prompt": "Telemetry context... Radio:", "completion": "Box this lap for hards"}
```

**Post-Race Dataset:**
**Post-Race Dataset:**
- **input**: JSON telemetry data (laps, positions, pit stops, stints, car data)
- **output**: Professional engineering debrief (9-section structured analysis)
- **metadata**: race details, driver, timestamp

## Model Fine-tuning

Fine-tune an LLM using QLoRA for efficient training with reduced memory requirements.

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

Post-race analysis dataset:
```bash
python fine_tune_granite_qlora.py \
    --model "ibm-granite/granite-3b-instruct" \
    --dataset "post_race_training_data_2024.jsonl" \
    --output "./granite_f1_postrace_finetuned" \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

Live race radio dataset:
```bash
python fine_tune_granite_qlora.py \
    --model "ibm-granite/granite-3b-instruct" \
    --dataset "f1_dataset_combined_filtered.jsonl" \
    --output "./granite_f1_live_finetuned" \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

## Arguments

- `--model`: Model name or path (default: `mistralai/Mistral-7B-Instruct-v0.2`)
- `--dataset`: Path to JSONL dataset (default: `post_race_training_data_2024.jsonl` or `post_race_combined.jsonl` for all seasons)
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

The dataset is in JSONL format with `input`, `output`, and `metadata` fields:

```json
{
  "input": "{\"laps\": [{\"lap\": 1, \"time\": 97.284}, ...], \"positions\": [...], \"pit_stops\": [...], \"stints\": [...], \"car_data\": [...]}",
  "output": "**Charles Leclerc - 2025 Bahrain Grand Prix**\n\n### 1. Overall Performance and Result\n...",
  "metadata": {"year": 2025, "gp": "Bahrain", "driver_abbr": "LEC", "driver_name": "Charles Leclerc", "num_events": 91, "timestamp": "2026-02-12 21:20:03"}
}
```

**Note:** You may need to adapt the training script to use `input`/`output` instead of `prompt`/`completion`.

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

# Use the model for post-race analysis
telemetry_json = '{"laps": [{"lap": 1, "time": 95.2}, ...], "positions": [...], "pit_stops": [...]}'
prompt = f"Analyze this F1 race telemetry:\n{telemetry_json}"
inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt")
outputs = model.generate(**inputs, max_length=2048)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)  # Should output a structured engineering debrief
```

## Tips

1. **Start with 3B model** for faster iteration
2. **Monitor GPU memory** - reduce batch size if OOM errors occur
3. **Use gradient accumulation** to simulate larger batch sizes
4. **Check TensorBoard logs** for training progress: `tensorboard --logdir ./granite_f1_finetuned/logs`
