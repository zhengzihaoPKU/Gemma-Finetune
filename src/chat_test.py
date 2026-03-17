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

def build_pipeline(merged_model, tokenizer):
    pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
    pipe_base = pipeline("text-generation", model="google/gemma-3-270m-it", device_map="auto")
    return pipe, pipe_base

if __name__ == "__main__":

    merged_model = load_merged_model()
    tokenizer = load_merged_tokenizer();

    pipe, pipe_base = build_pipeline(
        merged_model = merged_model,
        tokenizer = tokenizer
    )

    # Test a prompt
    text_to_translate = "Let's go to the beach."  #@param {type:"string"}
    inference_messages = [
        {"role": "system", "content": "Translate this text to emoji: "},
        {"role": "user", "content": text_to_translate}
    ]
    prompt = tokenizer.apply_chat_template(inference_messages, tokenize=False, add_generation_prompt=True)
    output = pipe(prompt, max_new_tokens=128)
    output_base = pipe_base(prompt, max_new_tokens=128)
    model_output = output[0]['generated_text'][len(prompt):].strip()
    model_output_base = output_base[0]['generated_text'][len(prompt):].strip()
    print(f"\nFine-tuned model output: {model_output}")
    print(f"\nBase model output: {model_output_base}")