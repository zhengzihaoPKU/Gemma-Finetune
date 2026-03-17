from transformers import (
    Trainer,
    TrainingArguments
)

def trainer_config_setup(args):
    """
    setup the config of trainer
    """
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        learning_rate=5e-5,                               # Learning rate,
        lr_scheduler_type="constant",                     # Use constant learning rate scheduler
        gradient_checkpointing=False,                     # Use gradient checkpointing to save memory
        optim="adamw_torch_fused",                        # Use fused adamw optimizer
        report_to="tensorboard",                          # Report metrics to tensorboard
        weight_decay=0.01,   
    )
    return training_args

def trainer_setup(model, training_args, dataset, tokenizer, data_collator):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    return trainer
