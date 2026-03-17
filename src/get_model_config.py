from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    with open("./configs/model_arch.json", 'w') as f:
        print(model, file=f)
    with open("./configs/model_config.json", 'w') as f:
        print(model.config, file=f)
    with open("./configs/tokenizer_arch.json", 'w') as f:
        print(tokenizer, file=f)
    print('okay.')