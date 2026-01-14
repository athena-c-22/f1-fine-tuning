import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

base_model_id = "ibm-granite/granite-3b-code-instruct-128k"  # base Granite model
adapter_path = "./granite_f1_finetuned"  # folder with fine-tuned adapter

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)  # Load from adapter path (has same tokenizer)

print("Loading base model with quantization (matching training setup)...")
# Load with same quantization config as training
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
    dtype=torch.float16,
    trust_remote_code=True,
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("✓ Model loaded successfully!")

# Format prompt using the model's chat template if available
prompt_text = "Telemetry: speed 190.4, rpm 9543.7, throttle 57.0, brake 25.2. Advice:"

# Try to use chat template if available
if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
else:
    # Fallback format for Granite/Mistral style
    formatted_prompt = f"<s>[INST] {prompt_text} [/INST]"

print(f"Prompt: {prompt_text}")
print(f"Formatted: {formatted_prompt[:100]}...")
print()

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*60)
print("Response:")
print("="*60)
print(response)
print("="*60)

# Note: merge_and_unload() may not work with quantized models
# If you need a merged model, you might need to load without quantization first
print("\n⚠ Note: merge_and_unload() may not work with 4-bit quantized models.")
print("   The adapter model works fine as-is for inference.")
print("   If you need a merged model, load base model without quantization first.")
