#!/usr/bin/env python3
from transformers import T5Tokenizer

for term in T5Tokenizer.from_pretrained("t5-base").get_vocab().keys():
    print(term)

