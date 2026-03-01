# F1 Fine-Tuning

Fine-tune large language models on Formula 1 telemetry and team radio data to create AI race engineers and analysts.

## Overview

This project provides tools to:
1. **Build training datasets** from real F1 telemetry and team radio (via the [OpenF1 API](https://openf1.org/))
2. **Filter datasets** to remove noise, gibberish, and non-informative content
3. **Fine-tune** an IBM Granite (or compatible instruct) model using QLoRA for memory-efficient training
4. **Run inference** with the resulting fine-tuned model

Two dataset types are supported:

| Script | Dataset type | Input | Output |
|---|---|---|---|
| `build_f1_race_engineer_dataset.py` | Race engineer | Telemetry + team radio (Whisper) | Telemetry → radio advice pairs |
| `build_f1_analyst_dataset.py` | Race analyst | Telemetry + Gemini LLM | Telemetry → race debrief report |

---

## Repository Structure

```
f1-fine-tuning/
├── build_f1_race_engineer_dataset.py  # Build dataset from telemetry + radio transcripts
├── build_f1_analyst_dataset.py        # Build dataset from telemetry + LLM-generated analyses
├── filter_dataset.py                  # Filter datasets (gibberish, non-English, conversational)
├── fine_tune_granite_qlora.py         # QLoRA fine-tuning script
├── load_finetuned_model.py            # Load and run inference on the fine-tuned model
├── requirements_finetune.txt          # Python dependencies
├── FINETUNE_README.md                 # Additional fine-tuning reference
└── f1_dataset_*.jsonl                 # Generated datasets (not tracked in git)
```

---

## Installation

```bash
pip install -r requirements_finetune.txt
```

For the analyst dataset builder, also install the Gemini SDK and dotenv:

```bash
pip install google-genai python-dotenv
```

For the race engineer dataset builder, also install Whisper and its dependencies:

```bash
pip install openai-whisper pandas
```

> **Note:** Whisper requires [ffmpeg](https://ffmpeg.org/download.html) to be installed and available on your `PATH`.

---

## Step 1: Build the Dataset

### Option A — Race Engineer Dataset (Telemetry + Team Radio)

Pairs real telemetry windows with transcribed team radio messages using [OpenAI Whisper](https://github.com/openai/whisper).

```bash
python build_f1_race_engineer_dataset.py
```

**Key configuration** (edit at the top of the file):

| Variable | Default | Description |
|---|---|---|
| `YEARS` | `[2025]` | F1 seasons to process |
| `SESSION_TYPE` | `"Race"` | Session type (`Race`, `Qualifying`, etc.) |
| `TELEMETRY_WINDOW_SECONDS` | `30` | Seconds of telemetry before each radio message |
| `WHISPER_MODEL` | `"base"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `OUTPUT_FILE` | `"f1_dataset.jsonl"` | Output file path |
| `CLEANUP_AUDIO_FILES` | `False` | Delete downloaded audio after transcription |

The output JSONL format is:
```json
{"prompt": "Telemetry: speed 290.1, rpm 11543.2, throttle 98.0, brake 0.0. Advice:", "completion": "Box this lap, box box box."}
```

### Option B — Race Analyst Dataset (Telemetry + LLM Analysis)

Uses the OpenF1 API to fetch full race telemetry, then calls Google Gemini to generate professional race debrief reports.

**Prerequisites:** Set your Gemini API key in a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

```bash
python build_f1_analyst_dataset.py
```

**Key configuration** (in `CONFIG` dict at the top of the file):

| Key | Default | Description |
|---|---|---|
| `years` | `[2023, 2024, 2025]` | F1 seasons to process |
| `max_races_per_year` | `10` | Maximum races to process per season |
| `drivers` | `None` | Specific driver numbers to process, or `None` for all |
| `llm.model` | `"gemini-3-flash-preview"` | Gemini model name |
| `jsonl_output` | `"f1_telemetry_analysis.jsonl"` | Output file path |

The output JSONL format is:
```json
{
  "input": "{...telemetry JSON...}",
  "output": "Subject: Dominant Victory — Strategic Masterclass...",
  "metadata": {"year": 2024, "grand_prix": "Monaco", "driver_name": "Max Verstappen", ...}
}
```

---

## Step 2: Filter the Dataset

Remove low-quality entries (gibberish transcriptions, non-English text, purely conversational messages) before training.

```bash
python filter_dataset.py
```

Edit the `main()` function to point at the correct input files. By default it looks for:
- `f1_dataset_2023.jsonl` → `f1_dataset_2023_filtered.jsonl`
- `f1_dataset_2024.jsonl` → `f1_dataset_2024_filtered.jsonl`
- `f1_dataset_2025.jsonl` → `f1_dataset_2025_filtered.jsonl`

If all three filtered files exist, they are automatically combined into `f1_dataset_combined_filtered.jsonl`.

Filtering criteria:
- **Gibberish**: low ASCII ratio, low letter ratio, excessive special characters
- **Non-English**: detected by [langid](https://github.com/saffsd/langid.py) with high confidence
- **Purely conversational**: no technical F1 keywords and only acknowledgements/pleasantries

---

## Step 3: Fine-Tune the Model

Fine-tune IBM Granite (or any compatible instruct model) using [QLoRA](https://arxiv.org/abs/2305.14314) for memory-efficient training.

### Basic Usage

```bash
python fine_tune_granite_qlora.py
```

### With Custom Parameters

```bash
python fine_tune_granite_qlora.py \
    --model "ibm-granite/granite-3b-code-instruct-128k" \
    --dataset "f1_dataset_combined_filtered.jsonl" \
    --output "./granite_f1_finetuned" \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `ibm-granite/granite-3b-code-instruct-128k` | HuggingFace model name or local path |
| `--dataset` | `f1_dataset_2024_filtered.jsonl` | Path to filtered JSONL dataset |
| `--output` | `./granite_f1_finetuned` | Output directory for model weights |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Per-device training batch size |
| `--learning-rate` | `2e-4` | Learning rate |

### Supported Models

| Model | Parameters | VRAM Required |
|---|---|---|
| `ibm-granite/granite-3b-code-instruct-128k` | 3B | ~8–12 GB |
| `ibm-granite/granite-7b-instruct` | 7B | ~12–16 GB |
| `ibm-granite/granite-8b-instruct` | 8B | ~16–20 GB |

### QLoRA Configuration

The script uses 4-bit quantization (NF4) with the following defaults:

| Parameter | Value |
|---|---|
| Quantization | 4-bit NF4 |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Optimizer | `paged_adamw_8bit` |

---

## Step 4: Run Inference

Load the fine-tuned model and generate responses:

```bash
python load_finetuned_model.py
```

Or use it programmatically:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "ibm-granite/granite-3b-code-instruct-128k"
adapter_path = "./granite_f1_finetuned"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = "Telemetry: speed 190.4, rpm 9543.7, throttle 57.0, brake 25.2. Advice:"
inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Tips

- **Start with a smaller model** (3B) to iterate quickly, then scale up.
- **Reduce batch size** (`--batch-size 1` or `2`) if you encounter out-of-memory errors.
- **Monitor training** with TensorBoard: `tensorboard --logdir ./granite_f1_finetuned/logs`
- **CPU training** is significantly slower; a cloud GPU (e.g., Google Colab, Kaggle) is recommended.
- Training can be safely interrupted with `Ctrl+C` — checkpoints are saved automatically.
