from huggingface_hub import (
    ModelCard, 
    ModelCardData, 
    whoami,
)
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)

def load_merged_model_and_tokenizer():
    """load model and tokenizer"""
    merged_model_path = "results/merge"  
    merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    return merged_model, tokenizer

def set_hf_repo():
    """set hr_repo_id"""
    username = whoami()['name']
    hf_repo_id = f"{username}/gemma-3-270m-it-emoji-finetune"
    return hf_repo_id

def push_to_hub(merged_model, hf_repo_id, tokenizer):
    """puth model to huggingface hub"""
    repo_url = merged_model.push_to_hub(
        hf_repo_id, 
        create_repo=True, 
        commit_message="Upload model"
    )
    tokenizer.push_to_hub(hf_repo_id)
    return repo_url

if __name__ == "__main__":
    merged_model, tokenizer = load_merged_model_and_tokenizer()
    hf_repo_id = set_hf_repo()
    repo_url = push_to_hub(
        merged_model=merged_model, 
        hf_repo_id=hf_repo_id, 
        tokenizer=tokenizer,
    )
    card_data = ModelCardData(
        language=["zh"],
        license=["mit"],
        library_name=["pytorch"],
        tags=["text-classification"],
    )
    card = ModelCard(card_data.to_yaml())
    card.push_to_hub(hf_repo_id)
    print(f"Uploaded to {repo_url}")