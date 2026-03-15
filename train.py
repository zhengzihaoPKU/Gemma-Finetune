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
    dtype="auto",  # 可选：使用 auto dtype 以提高效率  
    device_map="cuda:0"    # 可选：如果可用，自动映射到 GPU  
) 
print('-'*10 + "model load" + '-'*10)

# set pad_token for Gemma
# because Gemma tokenizer has no padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load dataset
dataset = load_dataset("sjoerdbodbijl/text-to-emoji")

# 3. 定义数据处理函数：核心修复padding/truncation + 标签标准化
def preprocess_function(examples):
    # 【修复1】开启padding/truncation，强制统一长度
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",  # 填充到max_length
        truncation=True,       # 截断超长文本
        max_length=128,        # 按需设置，建议根据数据分布调整
        return_tensors="pt"    # 可选：直接返回PyTorch张量
    )
    
    # 【修复2】标准化labels：展平嵌套、转整数、过滤异常值
    labels = []
    for label in examples["label"]:
        # 展平嵌套列表（比如[1]→1，[[2]]→2）
        while isinstance(label, list) and len(label) == 1:
            label = label[0]
        # 转整数（处理字符串标签，如"0"→0，需提前定义label2id映射）
        try:
            label = int(label)
        except (ValueError, TypeError):
            # 替换异常标签为默认值（或过滤，按需调整）
            label = 0
        labels.append(label)
    
    # 赋值labels，确保是一维整数列表
    tokenized["labels"] = labels
    return tokenized

# 4. 应用处理函数（batched=True批量处理）
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,  # 批量处理提升效率
    remove_columns=dataset["train"].column_names  # 删除原始列，只保留处理后的张量列
)

print('-'*10 + "dataset load" + '-'*10)

# set lora config
lora_config = LoraConfig(  
    r=8,   # lora rank
    lora_alpha=32,   
    target_modules=[  
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attn layer
        "gate_proj", "up_proj", "down_proj"     # MLP layer
    ], 
    lora_dropout=0.05,
    task_type="CAUSAL_LM"  # 为了清晰起见显式指定
)
# get lora model
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    remove_unused_columns=False,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
)

trainer.train()