from transformers import (  
    AutoModelForCausalLM,   # for model load
    AutoTokenizer,   # for tokenizer load
)

def model_loader(args):
    # load model and tokenizer
    print('-'*10 + "Model Loading..." + '-'*10)
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(  
        model_name,
        dtype="auto",  # 可选：使用 auto dtype 以提高效率  
    ).to(args.device)
    print("OK!")
    return model