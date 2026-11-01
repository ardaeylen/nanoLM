from RegexTokenizer import RegexTokenizer
from dataset import load_text_dataset

special_characters = {"eos":"\n<|endoftext|>\n"}
dataset_path = "/Users/arda/Desktop/Projects/LLM/NanoGPTFromScratch/Tokenizer/tinystories_sample.txt"
SAMPLED_CHARS = 10_000_000
VOCAB_SIZE =8192
SEPERATOR="<|endoftext|>"

dataset = load_text_dataset(dataset_path=dataset_path, sampled_chars=SAMPLED_CHARS)

stories = dataset.split(SEPERATOR)[:-1] # Do not take last unfinished story.

text = "\n".join(stories)                      # separators removed: BPE never sees "endoftext"
assert "<|endoftext|>" not in text

tokenizer = RegexTokenizer(special_tokens=["<|endoftext|>"])
tokenizer.train(text, VOCAB_SIZE)
tokenizer.save("tokenizer_8k.json")
print(f"{len(stories)} stories, {len(tokenizer.merges)} merges, vocab_size {tokenizer.vocab_size}, {tokenizer.special_tokens}")