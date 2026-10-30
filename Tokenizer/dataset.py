from datasets import load_dataset
from tqdm import tqdm
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
with open("tinystories_sample.txt", "w", encoding="utf-8") as f:
    for row in tqdm(ds):
        f.write(row["text"].strip() +"\n<|endoftext|>\n")