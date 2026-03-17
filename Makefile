model-clean:
	rm -rf ./logs/*

config-clean:
	rm -rf ./configs/*

tensorboard:
	tensorboard --logdir=./logs/runs/

train:
	bash scripts/lora.sh

chat_test:
	bash scripts/chat_test.sh