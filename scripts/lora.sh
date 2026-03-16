python train.py \
--model_name google/gemma-3-270m-it \
--dataset_name kr15t3n/text2emoji \
--device cuda:0 \
--lora_rank 16 \
--lora_alpha 32 \
--per_device_train_batch_size 4 \
--num_train_epochs 3 \