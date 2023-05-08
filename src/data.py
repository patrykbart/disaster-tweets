def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)