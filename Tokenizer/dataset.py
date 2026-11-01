from datasets import load_dataset
from tqdm import tqdm

def load_hf_dataset(name):
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    return ds
def seperate_dataset_sentences_with_special_token(ds, new_name, special_token=f"\n<|endoftext|>\n"):
    try:
        with open(new_name, "w", encoding="utf-8") as f:
            for row in tqdm(ds):
                f.write(row["text"].strip() + special_token)
    except Exception as e:
        print(e)

def load_text_dataset(dataset_path, sampled_chars):
    try:
        txt_file = """"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            txt_file = f.read(sampled_chars)
    except Exception as e:
        print(e)

    return txt_file

import numpy as np

def pretokenize(tok, src_path, dst_path, eot="<|endoftext|>", flush_every=10_000_000):
    assert tok.vocab_size <= 65536
    eot_id = tok.special_tokens[eot]
    buffer, story_lines = [], []
    with open(src_path, encoding="utf-8") as src, open(dst_path, "wb") as dst:
        for line in tqdm(src):                                   # reads one line at a time
            if line.rstrip("\n") == eot:
                story = "".join(story_lines).rstrip("\n")
                buffer.extend(tok.encode(story))
                buffer.append(eot_id)                      # EOT inserted as an id, never encoded as text
                story_lines = []
                if len(buffer) >= flush_every:
                    np.array(buffer, dtype=np.uint16).tofile(dst)
                    buffer = []
            else:
                story_lines.append(line)
        np.array(buffer, dtype=np.uint16).tofile(dst)      # last partial buffer