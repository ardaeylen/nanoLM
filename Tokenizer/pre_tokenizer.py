import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tokenizer"))
from RegexTokenizer import RegexTokenizer
from dataset import pretokenize

tokenizer = RegexTokenizer.load("tokenizer_8k.json")        # RegexTokenizer, not TokenizerBase (see bug 1)
os.makedirs("data", exist_ok=True)
pretokenize(tokenizer, "Tokenizer/tinystories_sample.txt", "data/tinystories_8k.bin")

data = np.memmap("data/tinystories_8k.bin", dtype=np.uint16, mode="r")
eot_id = tokenizer.special_tokens["<|endoftext|>"]
print(f"{len(data):,} tokens, max id {data.max()}, {(data == eot_id).sum():,} EOTs (expect 2,119,719)")
print(tokenizer.decode(data[:200].tolist()))