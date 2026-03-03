"""
Fine-tune IBM Granite 3B using QLoRA technique for F1 Race Engineer dataset.
[3B MODEL VARIANT - Run alongside 8B for comparison]

This script uses QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning
with reduced memory requirements. 3B model trains faster but has less capacity.
"""

import os
import json
import torch
import inspect
import warnings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
try:
    from trl import DataCollatorForCompletionOnlyLM
    _USE_TRL_COLLATOR = True
except ImportError:
    _USE_TRL_COLLATOR = False
# Granite uses special tokens (<|start_of_role|>, <|end_of_role|>) that TRL's collator
# fails to locate as a subsequence → all labels stay -100 → loss=0. Use manual masking.
_USE_TRL_COLLATOR = False
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import argparse

# Suppress PyTorch checkpoint warnings (they're just informational)
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*use_cache.*")


# Configuration - 3B Model Variant
# Training 3B and 8B simultaneously to compare performance
# 3B: Faster training, less memory, good baseline
# 8B: Better reasoning, more context understanding
MODEL_NAME = "ibm-granite/granite-4.0-micro"  
DATASET_PATH = "f1_dataset_2024_filtered.jsonl"
OUTPUT_DIR = "./granite_f1_finetuned_3b"  # Separate output directory
MAX_SEQ_LENGTH = 2048  # Post-race debriefs are long (JSON input + full analysis)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3  # Faster training (~4-5 hours vs 8B's 15 hours)
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.05


def load_and_prepare_dataset(dataset_path: str, tokenizer):
    """Load JSONL dataset and format for instruction fine-tuning."""
    print(f"Loading dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load JSONL file
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    
    print(f"Loaded {len(data)} examples")
    
    if len(data) == 0:
        raise ValueError("No valid examples found in dataset")
    
    # Format for instruction following
    # Store prompt and completion separately so we can compute the exact
    # prompt token length for label masking (avoids fragile token-search approach)
    formatted_data = []
    for item in data:
        # Support both formats: prompt/completion (race engineer) and input/output (post-race)
        prompt = item.get('prompt', item.get('input', ''))
        completion = item.get('completion', item.get('output', ''))

        if not prompt or not completion:
            continue

        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            try:
                # Full text: system + user + assistant response
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                # Prompt-only text: system + user + assistant header (no response)
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True
                )
            except Exception:
                full_text = f"<s>[INST] {prompt} [/INST] {completion}</s>"
                prompt_text = f"<s>[INST] {prompt} [/INST] "
        else:
            full_text = f"<s>[INST] {prompt} [/INST] {completion}</s>"
            prompt_text = f"<s>[INST] {prompt} [/INST] "

        formatted_data.append({"text": full_text, "prompt": prompt_text})
    
    if len(formatted_data) == 0:
        raise ValueError("No valid formatted examples created")
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize prompt-only to get exact boundary length for label masking.
    # This is more reliable than searching for template token IDs, which breaks
    # when truncation cuts the assistant marker out of the sequence.
    def tokenize_function(examples):
        # Tokenize full sequences with truncation
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
            return_tensors=None,
        )
        if not _USE_TRL_COLLATOR:
            # Tokenize prompt-only (no truncation) to find exact response start position
            prompt_tokenized = tokenizer(
                examples["prompt"],
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            all_labels = []
            skipped = 0
            for input_ids, prompt_ids in zip(tokenized["input_ids"], prompt_tokenized["input_ids"]):
                labels = [-100] * len(input_ids)
                response_start = len(prompt_ids)
                if response_start < len(input_ids):
                    # Mask prompt, train on response tokens only
                    for j in range(response_start, len(input_ids)):
                        labels[j] = input_ids[j]
                else:
                    skipped += 1  # Prompt alone exceeds MAX_SEQ_LENGTH, skip
                all_labels.append(labels)
            if skipped > 0:
                print(f"  Skipped {skipped}/{len(tokenized['input_ids'])} examples (prompt > MAX_SEQ_LENGTH)")
            tokenized["labels"] = all_labels
        return tokenized

    # Diagnostic: verify label masking works on first example
    sample = formatted_data[0]
    sample_full_ids = tokenizer(sample["text"], truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
    sample_prompt_ids = tokenizer(sample["prompt"], truncation=False)["input_ids"]
    response_start = len(sample_prompt_ids)
    trainable = max(0, len(sample_full_ids) - response_start)
    print(f"Label masking check — example 0: prompt={response_start} tokens, "
          f"full={len(sample_full_ids)} tokens, trainable={trainable} response tokens")
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Diagnostic: check how many examples have non-masked labels
    if not _USE_TRL_COLLATOR and 'labels' in tokenized_dataset.column_names:
        sample_labels = tokenized_dataset[0]['labels']
        non_masked = sum(1 for l in sample_labels if l != -100)
        print(f"Label masking check — example 0: {non_masked}/{len(sample_labels)} tokens are trainable (non -100)")
        if non_masked == 0:
            print("ERROR: All labels are -100! Response template not found. Check token IDs above.")

    # For causal LM, labels should be the same as input_ids
    # The DataCollatorForLanguageModeling will handle this, but we can also set it explicitly
    # Let's not add labels here - let the data collator handle it to avoid nesting issues
    # The data collator will automatically create labels from input_ids
    
    return tokenized_dataset


def find_target_modules(model):
    """Auto-detect target modules for LoRA based on model architecture."""
    # Common module names for different architectures
    common_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",  # MLP (Mistral/Llama)
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",  # GPT-style
    ]
    
    # Get all module names from the model
    module_names = set()
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            module_names.add(name.split('.')[-1])
    
    # Find matching modules
    target_modules = [m for m in common_modules if m in module_names]
    
    if not target_modules:
        # Fallback: try to find any linear layers in attention/MLP
        print("Warning: Could not auto-detect target modules, using default")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    print(f"Target modules for LoRA: {target_modules}")
    return target_modules


def setup_model_and_tokenizer(model_name: str):
    """Setup model with QLoRA configuration."""
    print(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization
        print("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,  # Changed from torch_dtype (deprecated)
            trust_remote_code=True,
        )
        
        # Prepare model for k-bit training
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        
        # Disable cache for training (required for gradient checkpointing)
        if hasattr(model, 'config'):
            model.config.use_cache = False
        
        # Auto-detect target modules
        target_modules = find_target_modules(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=target_modules,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        print("Applying LoRA adapters...")
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error setting up model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    global MAX_SEQ_LENGTH
    parser = argparse.ArgumentParser(description="Fine-tune model with QLoRA")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length (reduce to save GPU memory)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("F1 Race Engineer Model Fine-tuning with QLoRA")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 60)
    print()
    
    # Setup model and tokenizer
    try:
        model, tokenizer = setup_model_and_tokenizer(args.model)
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the model name is correct")
        print("2. Ensure you have internet connection (for HuggingFace models)")
        print("3. Check if you have enough GPU memory")
        print("4. Try a smaller model or reduce batch size")
        return
    
    # Override MAX_SEQ_LENGTH from CLI arg
    MAX_SEQ_LENGTH = args.max_seq_length
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")

    # Load and prepare dataset
    try:
        dataset = load_and_prepare_dataset(args.dataset, tokenizer)
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        return
    
    # Split dataset (90% train, 10% validation)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"\nTrain examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    print()
    
    # Check dataset structure
    print("Checking dataset structure...")
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Sample input_ids type: {type(sample.get('input_ids'))}")
    print(f"Sample input_ids length: {len(sample.get('input_ids', []))}")
    if 'labels' in sample:
        print(f"Sample labels type: {type(sample.get('labels'))}")
        if _USE_TRL_COLLATOR:
            # TRL collator creates labels itself — remove pre-existing ones
            print(f"⚠ Removing pre-existing labels (TRL collator will create them)...")
            train_dataset = train_dataset.remove_columns(['labels']) if 'labels' in train_dataset.column_names else train_dataset
            eval_dataset = eval_dataset.remove_columns(['labels']) if 'labels' in eval_dataset.column_names else eval_dataset
        else:
            print(f"✓ Using pre-computed masked labels")
    print()
    
    # Only compute loss on assistant response tokens, not the input JSON
    if _USE_TRL_COLLATOR:
        response_template = "<|start_of_role|>assistant<|end_of_role|>"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )
    else:
        # Labels already masked in tokenize_function; just pad the batch
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )
    
    # Test the data collator with a sample
    print("Testing data collator with sample batch...")
    try:
        sample_batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
        collated = data_collator(sample_batch)
        print(f"✓ Data collator test passed")
        print(f"  Batch keys: {list(collated.keys())}")
        if 'input_ids' in collated:
            print(f"  Input shape: {collated['input_ids'].shape}")
        if 'labels' in collated:
            print(f"  Labels shape: {collated['labels'].shape}")
            print(f"  Labels dtype: {collated['labels'].dtype}")
    except Exception as e:
        print(f"✗ Data collator test failed: {e}")
        print("  This indicates an issue with dataset formatting")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Training arguments
    # Check which parameter name is valid (eval_strategy vs evaluation_strategy)
    sig = inspect.signature(TrainingArguments.__init__)
    eval_param_name = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"
    
    # Check if tensorboard is available
    try:
        import tensorboard
        report_to = "tensorboard"
    except ImportError:
        print("⚠ TensorBoard not installed. Logging to console only.")
        print("  Install with: pip install tensorboard")
        report_to = "none"
    
    training_kwargs = {
        "output_dir": args.output,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": args.learning_rate,
        "fp16": True,  # Mixed precision training
        "bf16": False,  # Use fp16, not bf16
        "logging_steps": 1,  # Log every step to see progress immediately
        "save_steps": 50,  # Save more frequently for safety
        "save_strategy": "steps",  # Save based on steps
        "eval_steps": 50,  # Evaluate more frequently
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "warmup_steps": 50,
        "report_to": report_to,
        "logging_dir": f"{args.output}/logs",
        "optim": "paged_adamw_8bit",  # Memory efficient optimizer
        "dataloader_pin_memory": False,  # Can cause issues on some systems
        "remove_unused_columns": False,  # Keep all columns
    }
    
    # Use the correct parameter name based on transformers version
    training_kwargs[eval_param_name] = "steps"
    
    training_args = TrainingArguments(**training_kwargs)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    print("Note: Training with 4-bit quantization can be slower but uses less memory.")
    print("You can interrupt with Ctrl+C - checkpoints are saved automatically.")
    print()
    steps_per_epoch = len(train_dataset) // (args.batch_size * GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * args.epochs
    print(f"Dataset: {len(train_dataset)} training examples")
    print(f"Effective batch size: {args.batch_size} × {GRADIENT_ACCUMULATION_STEPS} = {args.batch_size * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Steps per epoch: ~{steps_per_epoch}")
    print(f"Total training steps: ~{total_steps}")
    print()
    
    # Check if using CPU or GPU
    import torch
    if torch.cuda.is_available():
        print("✓ GPU detected - training will be faster")
        print("⚠ First step can take 30-120 seconds (CUDA initialization, compilation)")
        print("   Subsequent steps: ~1-5 seconds each")
        estimated_time = total_steps * 3  # ~3 seconds per step on GPU
        print(f"   Estimated total time: ~{estimated_time // 60} minutes")
    else:
        print("⚠ CPU training detected - this will be MUCH slower!")
        print("   First step: 60-180 seconds (model compilation)")
        print("   Subsequent steps: 30-90 seconds each (CPU is slow for neural networks)")
        estimated_time = total_steps * 60  # ~60 seconds per step on CPU
        print(f"   Estimated total time: ~{estimated_time // 60} minutes ({estimated_time // 3600} hours)")
        print()
        print("💡 Tips for CPU training:")
        print("   - Consider reducing batch size: --batch-size 1 or 2")
        print("   - Reduce MAX_SEQ_LENGTH to 256 or 128")
        print("   - Use fewer epochs: --epochs 1 or 2")
        print("   - Consider using a cloud GPU (Google Colab, Kaggle, etc.)")
    print()
    
    try:
        import time
        print("Starting training loop...")
        start_time = time.time()
        
        # Resume from checkpoint if one exists, otherwise start fresh
        checkpoints = [d for d in os.listdir(args.output) if d.startswith("checkpoint-")] if os.path.exists(args.output) else []
        resume = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1] if checkpoints else None
        if resume:
            print(f"Resuming from checkpoint: {resume}")
        trainer.train(resume_from_checkpoint=os.path.join(args.output, resume) if resume else None)
        
        # Save final model
        print(f"\nSaving model to {args.output}...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output)
        
        print("=" * 60)
        print("Training complete!")
        print(f"Model saved to: {args.output}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user (Ctrl+C)")
        print("Saving checkpoint...")
        try:
            trainer.save_model()
            tokenizer.save_pretrained(args.output)
            print(f"✓ Checkpoint saved to: {args.output}")
            print("You can resume training later or use this checkpoint.")
        except Exception as save_error:
            print(f"⚠ Warning: Could not save checkpoint: {save_error}")
        print("\nTraining stopped.")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n✗ GPU Out of Memory Error: {e}")
            print("\nTroubleshooting tips:")
            print("1. Reduce batch size: --batch-size 2 or --batch-size 1")
            print("2. Increase gradient accumulation: modify GRADIENT_ACCUMULATION_STEPS")
            print("3. Reduce MAX_SEQ_LENGTH in the script")
            print("4. Use a smaller model")
        else:
            print(f"\n✗ Training error: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"\n✗ Unexpected error during training: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

