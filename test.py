from transformers import (
    AutoModelForCausalLM,   # for model load
    AutoTokenizer,   # for tokenizer load
)
from peft import AutoPeftModelForCausalLM

lora_model = AutoPeftModelForCausalLM.from_pretrained("/home/zihao/Gemma-finetune/results/checkpoint-800").to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained("/home/zihao/Gemma-finetune/results/checkpoint-800")
prompt = "translate 'Feeling happy' to emoji\n"
input_data = tokenizer(prompt, return_tensors="pt").to('cuda')
output = lora_model.generate(
    input_data["input_ids"],
    do_sample=True,
    max_new_tokens=50, 
    top_p=0.9,              # 核采样阈值
    temperature=0.7,        # 控制随机性（可选）
    pad_token_id=tokenizer.eos_token_id
)
generated_text = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)  # 转为文本

print(generated_text)
# input_data = tokenizer(prompt, return_tensors="pt").to('cuda')
# output = lora_model(input_data['input_ids'])
# print(output)