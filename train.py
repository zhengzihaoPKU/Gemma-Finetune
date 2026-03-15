from transformers import (  
    AutoModelForCausalLM,   # for model load
    AutoTokenizer,   # for tokenizer load
    Trainer,   # Train tools
    TrainingArguments,   # Train tools
    DataCollatorForLanguageModeling  # Dataloader
)  
from peft import LoraConfig, get_peft_model  
from datasets import load_dataset

# load model and tokenizer
model_name = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(  
    model_name,   
    torch_dtype="auto",  # 可选：使用 auto dtype 以提高效率  
    device_map="cuda:0"    # 可选：如果可用，自动映射到 GPU  
) 
print('-'*10 + "model load" + '-'*10)

# set pad_token for Gemma
# because Gemma tokenizer has no padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load dataset
dataset = load_dataset("json", data_files="./data/emoji_dataset.json")  
print('-'*10 + "dataset load" + '-'*10)

# 可选：如果序列很长，进行预分词和截断（Trainer 可以处理原始文本，但这样更明确）  
# def tokenize_function(examples):  
#     return tokenizer(examples["text"], truncation=True, max_length=512)  
# dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)  

# set lora config
lora_config = LoraConfig(  
    r=8,   # lora rank
    lora_alpha=32,   
    target_modules=[  
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attn layer
        "gate_proj", "up_proj", "down_proj"     # MLP layer
    ], 
    lora_dropout=0.05,
    init_lora_weights = "kaiming",
    task_type="CAUSAL_LM"  # 为了清晰起见显式指定
)
# get lora model
model = get_peft_model(model, lora_config)

# set training configs
training_args = TrainingArguments(  
    output_dir="./gemma-emoji",  
    num_train_epochs=3,  
    per_device_train_batch_size=4,  
    save_steps=100,  
    logging_steps=10,      # 可选：更频繁地记录日志
    evaluation_strategy="no",  # 如果你有 eval_dataset，请在此添加  
    # group_by_length=True,  # 可选：将相似长度分组以提高效率  
    # max_steps=-1,          # 可选：运行完整的 epoch  
)

# 关键：适用于 CLM 的正确整理器 (collator)  
data_collator = DataCollatorForLanguageModeling(  
    tokenizer=tokenizer,   
    mlm=False  # 因果语言模型 (Causal LM)，非掩码  
)  

# train
trainer = Trainer(  
    model=model,   
    args=training_args,   
    train_dataset=dataset["train"],
    tokenizer=tokenizer,       # 如果未预分词，则启用自动分词  
    data_collator=data_collator  
)  

trainer.train()