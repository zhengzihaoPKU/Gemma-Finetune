from transformers import (  
    AutoModelForCausalLM,   # for model load
    AutoTokenizer,   # for tokenizer load
)

def tokenizer_loader(args):
    model_name = args.model_name
    print('-'*10 + "Tokenizer Loading..." + '-'*10)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # set pad_token for Gemma
    # because Gemma tokenizer has no padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("OK!")
    return tokenizer