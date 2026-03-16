from transformers import (  
    AutoModelForCausalLM,   # for model load
    AutoTokenizer,   # for tokenizer load
    Trainer,   # Train tools
    TrainingArguments,   # Train tools
    DataCollatorForLanguageModeling  # Dataloader
)
from utils.dataset_loader import (
    dataset_loader,
    dataset_preprocess
)
from utils.model_loader import model_loader
from utils.tokenizer_loader import tokenizer_loader
from utils.lora import (
    lora_config_setup,
    get_lora_model
)
from utils.data_collator import data_collator_setup
from utils.trainer import (
    trainer_config_setup,
    trainer_setup
)
import argparse

def get_args():
    parser = argparse.ArgumentParser(prog='ProgramName', description='LoRA Finetune', epilog='Text at the bottom of help')
    parser.add_argument('--model_name', default='', type=str, help='the model name in Huggingface')
    parser.add_argument('--dataset_name', default='', type=str, help='the dataset name in Huggingface')
    parser.add_argument('--device', default='cuda:0', type=str, help='GPU device')
    
    # args of lora config
    parser.add_argument('--lora_rank', default=8, type=int, help='rank of LoRA')
    parser.add_argument('--lora_alpha', default=32, type=int, help='alpha of LoRA')
    parser.add_argument('--lora_dropout', default=0.05, type=float, help='dropout rate of LoRA')
    parser.add_argument('--lora_tasktype', default="CAUSAL_LM", type=str, help='task type of LoRA')

    # args of transformer.trainer
    parser.add_argument('--output_dir', default="./results", type=str, help='model save path')
    parser.add_argument('--per_device_train_batch_size', default=4, type=int, help='...')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='...')
    parser.add_argument('--logging_steps', default=10, type=int, help='...')

    # Run the parser and put the extracted data into an argparse.Namespace object
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # get argument
    args = get_args()
    print(args)

    # get model and tokenizer
    model = model_loader(args=args)
    tokenizer = tokenizer_loader(args=args)

    # get dataset and preporcess it
    dataset = dataset_loader(args=args)
    preprocessed_dataset = dataset_preprocess(dataset=dataset, tokenizer=tokenizer)

    # get lora model
    lora_config = lora_config_setup(args=args)
    model = get_lora_model(
        model = model,
        lora_config = lora_config
    )

    # get data collator
    data_collator = data_collator_setup(tokenizer=tokenizer)

    # set training config
    training_args = trainer_config_setup(args=args)
    trainer = trainer_setup(
        model = model,
        training_args = training_args,
        dataset = preprocessed_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
    )

    trainer.train()

    # # 关键：适用于 CLM 的正确整理器 (collator)
    # data_collator = DataCollatorForLanguageModeling(  
    #     tokenizer=tokenizer,   
    #     mlm=False  # 因果语言模型 (Causal LM)，非掩码  
    # )

    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     per_device_train_batch_size=4,
    #     num_train_epochs=1,
    #     remove_unused_columns=False,
    #     logging_steps=10,
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator
    # )

    # trainer.train()