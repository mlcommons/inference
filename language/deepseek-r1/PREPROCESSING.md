# DeepSeek-R1 Preprocessing

## Tokenization
```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length = 4096
```

## Prompt Template (GSM8K)
```
<|im_start|>system
You are a helpful assistant that thinks step by step.<|im_end|>
<|im_start|>user
{question}

Let's think about this step by step.<|im_end|>
<|im_start|>assistant
```

## Answer Extraction
```python
# Extract final answer
output = full_output.split('<|/thinking|>')[-1] if '<|/thinking|>' in full_output else full_output
answer = re.search(r'####\s*(\d+)', output)
final_answer = answer.group(1) if answer else output.strip()
```