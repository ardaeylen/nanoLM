# Round-trip, training, special token and save/load tests for TokenizerBase and RegexTokenizer.
# Run from anywhere: python tests/test_tokenizer.py
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_DIR = os.path.join(REPO_ROOT, "Tokenizer")
sys.path.insert(0, TOKENIZER_DIR)

import regex as re
from tokenizer_base import TokenizerBase
from RegexTokenizer import RegexTokenizer, GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN

DATA_PATH = os.path.join(TOKENIZER_DIR, "tiny_shakespeare.txt")
SPECIALS = ["<|endoftext|>", "<|pad|>"]


# Independent reference: plain chunked BPE without deduplication, written separately from the tokenizer classes.
# If RegexTokenizer.train ever disagrees with this, one of the optimizations broke something.
def reference_bpe(text, num_merges, initial_vocab_size=256, pattern=GPT2_SPLIT_PATTERN):
    chunks = [list(chunk.encode("utf-8")) for chunk in re.findall(pattern, text)]
    merges = {}
    for new_idx in range(initial_vocab_size, initial_vocab_size + num_merges):
        stats = {}
        for ids in chunks:
            for pair in zip(ids, ids[1:]):
                stats[pair] = stats.get(pair, 0) + 1
        if not stats:
            break
        top_pair = max(stats, key=stats.get)
        if stats[top_pair] < 2:
            break
        new_chunks = []
        for ids in chunks:
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == top_pair:
                    out.append(new_idx)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            new_chunks.append(out)
        chunks = new_chunks
        merges[top_pair] = new_idx
    return merges


results = []

def check(name, condition):
    try:
        ok, error = bool(condition()), ""
    except Exception as e:
        ok, error = False, f"  [{type(e).__name__}: {e}]"
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + error)


def main():
    if not os.path.exists(DATA_PATH):
        sys.exit(f"Missing {DATA_PATH} (it is gitignored, download Tiny Shakespeare first)")
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()
    train_text, test_text = text[:200_000], text[300_000:320_000]   # test text is never seen in training
    cases = ["the cat saw the dog and the bird", "a a a", "zq", "", "e", "Merhaba dünya, ğüşıöç 🙂",
             "  multiple   spaces\n\n\ttab", "I'm can't 12345 3.14", test_text]
    tmp = tempfile.mkdtemp()

    for cls in (TokenizerBase, RegexTokenizer):
        name = cls.__name__
        fresh = cls(min_frequency=3)
        check(f"{name}: fresh vocab_size 256, min_frequency 3", lambda: fresh.vocab_size == 256 and fresh.min_frequency == 3)
        check(f"{name}: untrained round-trip", lambda: all(fresh.decode(fresh.encode(s)) == s for s in cases))
        names = ["<|endoftext|>"]
        copied = cls(special_tokens=names)
        names.append("changed later")
        check(f"{name}: special_token_names is a copy", lambda: copied.special_token_names == ["<|endoftext|>"])

        data, size = (train_text[:20_000], 306) if cls is TokenizerBase else (train_text, 556)
        for specials in ([], SPECIALS):
            k = len(specials)
            tok = cls(special_tokens=specials)
            tok.train(data, size)
            check(f"{name} specials={k}: vocab_size == {size}", lambda: tok.vocab_size == size)
            check(f"{name} specials={k}: merges are ids 256..{size - k - 1}", lambda: sorted(tok.merges.values()) == list(range(256, size - k)))
            check(f"{name} specials={k}: special ids right after merges", lambda: list(tok.special_tokens.values()) == list(range(size - k, size)))
            check(f"{name} specials={k}: round-trip all cases", lambda: all(tok.decode(tok.encode(s)) == s for s in cases))
            check(f"{name} specials={k}: compresses, all ids in vocab",
                  lambda: (lambda ids: len(ids) < len(test_text.encode()) and all(i in tok.vocab for i in ids))(tok.encode(test_text)))
            before = (list(tok.merges.items()), dict(tok.special_tokens))
            tok.train(data, size)
            check(f"{name} specials={k}: retrain gives identical merges + ids", lambda: (list(tok.merges.items()), tok.special_tokens) == before)
            path = os.path.join(tmp, f"{name}_{k}.json")
            tok.save(path)
            loaded = cls.load(path)
            check(f"{name} specials={k}: save/load identical",
                  lambda: list(loaded.merges.items()) == list(tok.merges.items()) and loaded.vocab == tok.vocab
                  and loaded.special_tokens == tok.special_tokens and loaded._config() == tok._config()
                  and loaded.vocab_size == size and loaded.encode(test_text) == tok.encode(test_text))

    regex_tok = RegexTokenizer()
    regex_tok.train(train_text, 556)
    check("Regex: merges identical to independent reference BPE",
          lambda: list(regex_tok.merges.items()) == list(reference_bpe(train_text, 300).items()))
    regex_tok.train(train_text, 300)
    check("Regex: retrain smaller -> 44 merges", lambda: len(regex_tok.merges) == 44 and regex_tok.vocab_size == 300)

    early = RegexTokenizer(min_frequency=10**9, special_tokens=SPECIALS)
    early.train(train_text, 556)
    check("early stop: special ids follow the last real merge (no gap)",
          lambda: list(early.special_tokens.values()) == [256, 257] and early.vocab_size == 258)

    gpt4 = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN, special_tokens=SPECIALS)
    gpt4.train(train_text, 400)
    gpt4_path = os.path.join(tmp, "gpt4.json")
    gpt4.save(gpt4_path)
    gpt4_loaded = RegexTokenizer.load(gpt4_path)
    check("GPT-4 pattern + specials survive save/load",
          lambda: gpt4_loaded.pattern == GPT4_SPLIT_PATTERN and gpt4_loaded.special_tokens == gpt4.special_tokens
          and gpt4_loaded.encode(test_text) == gpt4.encode(test_text))

    def wrong_class_rejected():
        try:
            TokenizerBase.load(gpt4_path)
            return False
        except AssertionError:
            return True
    check("loading with the wrong class is rejected", wrong_class_rejected)

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
