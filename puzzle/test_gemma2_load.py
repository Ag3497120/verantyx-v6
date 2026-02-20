#!/usr/bin/env python3
"""
Test Gemma2-2B model loading and basic inference
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('[Test] Loading Gemma2-2B-IT...')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')
print(f'[Test] Tokenizer loaded: {len(tokenizer)} tokens')

model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-2-2b-it',
    torch_dtype=torch.float32,
    device_map='cpu'
)
print('[Test] Model loaded successfully')
print(f'[Test] Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')

# Quick test
prompt = 'Extract variables from: What is C(10,3)?'
inputs = tokenizer(prompt, return_tensors='pt')
print(f'[Test] Input tokens: {inputs.input_ids.shape}')

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=0.0)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'[Test] Output: {output_text[:200]}...')
print('[Test] ✓ Gemma2-2B working!')
