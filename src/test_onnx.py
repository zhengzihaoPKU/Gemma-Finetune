from transformers import AutoConfig, AutoTokenizer, GenerationConfig
import onnxruntime
import numpy as np

save_path = "./logs/onnx"

# Load config, processor, and model
config = AutoConfig.from_pretrained(save_path)
generation_config = GenerationConfig.from_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path)

model_file = "onnx/model.onnx"      #@param ["onnx/model.onnx", "onnx/model_fp16.onnx", "onnx/model_q4.onnx", "onnx/model_q4f16.onnx"]

model_path = f"{save_path}/{model_file}"
decoder_session = onnxruntime.InferenceSession(model_path)

## Set config values
num_key_value_heads = config.num_key_value_heads
head_dim = config.head_dim
num_hidden_layers = config.num_hidden_layers
eos_token_id = tokenizer.eos_token_id

# Prepare inputs
text_to_translate = "i love sushi"      # @param {type:"string"}
messages = [
  { "role": "system", "content": "Translate this text to emoji: " },
  { "role": "user", "content": text_to_translate },
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
batch_size = input_ids.shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}
position_ids = np.tile(np.arange(0, input_ids.shape[-1]), (batch_size, 1))

# 3. Generation loop
max_new_tokens = 8
generated_tokens = np.array([[]], dtype=np.int64)

for i in range(max_new_tokens):
  logits, *present_key_values = decoder_session.run(None, dict(
      input_ids=input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      **past_key_values,
  ))

  ## Update values for next generation loop
  input_ids = logits[:, -1].argmax(-1, keepdims=True)
  attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids, dtype=np.int64)], axis=-1)
  position_ids = position_ids[:, -1:] + 1

  for j, key in enumerate(past_key_values):
    past_key_values[key] = present_key_values[j]

  generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)

  if np.isin(input_ids, eos_token_id).any():
    break

# 4. Output result
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])