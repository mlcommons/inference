# Llama 3.1 8B Preprocessing

## Tokenization
```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length = 2048
```

## Prompt Template (CNN/DailyMail)
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize this article:

{article}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

## Verification
```bash
# Verify first token
python -c "print(tokenizer.encode('<|begin_of_text|>')[0])"
```