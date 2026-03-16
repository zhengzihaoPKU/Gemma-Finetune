import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

adapter_path = "results/checkpoint-951"                 # Choose which adapters to merge, otherwise defaults to latest
merged_model_path = "results/merge"              # Location of merged model directory

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load and merge the PEFT adapters onto the base model
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

# Save the merged model and its tokenizer
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Model merged and saved to {merged_model_path}. Final model vocabulary size: {model.config.vocab_size}")