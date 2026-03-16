python train.py \
--model_name google/gemma-3-270m \
--dataset_name sjoerdbodbijl/text-to-emoji \
--device cuda:0 \
--per_device_train_batch_size 128 \
--num_train_epochs 100 \