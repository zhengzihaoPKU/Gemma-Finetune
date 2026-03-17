import torch.onnx
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)

def load_merged_model():
    merged_model_path = "./logs/merge"  
    # Create Transformers inference pipeline
    merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto")
    return merged_model

def load_merged_tokenizer():
    merged_model_path = "./logs/merge" 
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    return tokenizer

if __name__ == "__main__":
    merged_model = load_merged_model()
    tokenizer = load_merged_model()
    merged_model.eval()

    dummy_input = tokenizer(
        "hello Hugging Face!",
        return_tensors = "pt",
        padding = "max_length",
        max_length = 
    )