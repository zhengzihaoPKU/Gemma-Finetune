from transformers import DataCollatorForLanguageModeling

def data_collator_setup(tokenizer):
    """The correct organizer for CLM"""
    data_collator = DataCollatorForLanguageModeling(  
        tokenizer=tokenizer,   
        mlm=False  # Causal LM，no mask
    )
    return data_collator