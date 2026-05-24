#!/usr/bin/env python3
"""
MLCommons Inference - Preprocessing Examples

This script demonstrates correct preprocessing for different models.
Based on actual implementations in the codebase.
"""

from transformers import AutoTokenizer
import pandas as pd


def preprocess_deepseek_r1(prompts, use_chat_template=True):
    """
    Preprocess prompts for DeepSeek-R1 model.
    
    Args:
        prompts: List of text prompts
        use_chat_template: Whether to use chat template (depends on backend)
    
    Returns:
        List of tokenized prompts
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1",
        revision="56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"
    )
    
    tokenized = []
    for prompt in prompts:
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                max_length=32768,
                truncation=True
            )
        else:
            tokens = tokenizer.encode(
                prompt,
                truncation=True,
                max_length=32768
            )
        tokenized.append(tokens)
    
    return tokenized


def preprocess_llama31_8b(articles):
    """
    Preprocess articles for Llama 3.1-8B summarization.
    
    Args:
        articles: List of articles to summarize
    
    Returns:
        List of tokenized prompts
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 8000
    
    # Template from prepare-calibration.py
    instruction_template = "Summarize the following news article in 128 tokens. Please output the summary only, without any other text.\n\nArticle:\n{input}\n\nSummary:"
    
    tokenized = []
    for article in articles:
        prompt = instruction_template.format(input=article)
        tokens = tokenizer.encode(prompt, max_length=8000, truncation=True)
        tokenized.append(tokens)
    
    return tokenized


def preprocess_llama2_70b(prompts, system_prompts=None):
    """
    Preprocess prompts for Llama 2-70B model.
    
    Args:
        prompts: List of user prompts
        system_prompts: Optional list of system prompts
    
    Returns:
        List of tokenized prompts
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-70b-chat-hf",
        use_fast=False
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Templates from processorca.py
    llama_prompt_system = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]"
    llama_prompt_no_system = "<s>[INST] {} [/INST]"
    
    tokenized = []
    for i, prompt in enumerate(prompts):
        if system_prompts and system_prompts[i]:
            formatted = llama_prompt_system.format(system_prompts[i], prompt)
        else:
            formatted = llama_prompt_no_system.format(prompt)
        
        tokens = tokenizer.encode(formatted, max_length=1024, truncation=True)
        tokenized.append(tokens)
    
    return tokenized


def create_dataset_format(prompts, tokenized_prompts, outputs=None):
    """
    Create dataset in expected format for MLCommons.
    
    Args:
        prompts: List of text prompts
        tokenized_prompts: List of tokenized prompts
        outputs: Optional list of expected outputs
    
    Returns:
        DataFrame in expected format
    """
    data = {
        'text_input': prompts,
        'tok_input': tokenized_prompts,
    }
    
    if outputs:
        data['output'] = outputs
    
    return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    # Example 1: DeepSeek-R1
    print("=== DeepSeek-R1 Example ===")
    deepseek_prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms."
    ]
    
    # With chat template (PyTorch/vLLM)
    deepseek_tokens = preprocess_deepseek_r1(deepseek_prompts, use_chat_template=True)
    print(f"Prompt 1 token count: {len(deepseek_tokens[0])}")
    
    # Without chat template (SGLang)
    deepseek_tokens_no_chat = preprocess_deepseek_r1(deepseek_prompts, use_chat_template=False)
    print(f"Prompt 1 token count (no chat): {len(deepseek_tokens_no_chat[0])}")
    
    # Example 2: Llama 3.1-8B
    print("\n=== Llama 3.1-8B Example ===")
    articles = [
        "The United Nations announced today a new climate initiative aimed at reducing global emissions by 50% by 2030. The plan includes partnerships with major corporations and governments worldwide."
    ]
    
    llama_tokens = preprocess_llama31_8b(articles)
    print(f"Article 1 token count: {len(llama_tokens[0])}")
    
    # Example 3: Create dataset
    print("\n=== Dataset Format Example ===")
    df = create_dataset_format(deepseek_prompts, deepseek_tokens)
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Save example
    # df.to_pickle("preprocessed_data.pkl")