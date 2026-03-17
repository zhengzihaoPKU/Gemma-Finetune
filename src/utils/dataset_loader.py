from datasets import load_dataset

def dataset_loader(args):
    """
    load dataset from Huggingface
    """
    dataset_name = args.dataset_name
    # load dataset
    print('-'*10 + "Dataset Loading..." + '-'*10)
    dataset = load_dataset(args.dataset_name)
    print("OK!")
    return dataset

def dataset_preprocess(dataset,tokenizer):
    # 可选：如果序列很长，进行预分词和截断（Trainer 可以处理原始文本，但这样更明确）  
    def tokenize_function(examples):  
        return tokenizer(examples["text"], truncation=True, max_length=512)  
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    return dataset