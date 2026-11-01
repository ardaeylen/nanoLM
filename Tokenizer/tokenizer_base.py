import json

class TokenizerBase:
    def __init__(self, min_frequency: int = 2):
        self.merges = {}
        self.min_frequency = min_frequency
        self.initial_vocab_size = 256
        self.special_tokens = {}
        self._build_vocab()

    def get_pair_counts(self, ids, stats=None, freq=1):
        stats = {} if stats is None else stats
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + freq
        return stats

    def merge(self, id_sequence, new_idx, pair):
        new_id_sequence = []
        num_ids = len(id_sequence)
        idx = 0
        while idx < num_ids:
            if idx <  num_ids - 1 and id_sequence[idx] == pair[0] and id_sequence[idx + 1] == pair[1]:
                new_id_sequence.append(new_idx)
                idx += 2
            else:
                new_id_sequence.append(id_sequence[idx])
                idx += 1
        return new_id_sequence


    def train(self, text, vocab_size):
        self.merges = {}
        ids = self.byte_encoder(text)
        vocab_idx = self.initial_vocab_size
        num_merges = vocab_size - self.initial_vocab_size
        merge_count = 0
        while merge_count < num_merges:
            current_stats = self.get_pair_counts(ids)
            if not current_stats: # fewer than 2 ids left, nothing to merge
                break
            top_pair = max(current_stats, key=current_stats.get)
            if current_stats[top_pair] < self.min_frequency:  # merging a pair that occurs once doesn't compress anything
                break
            ids = self.merge(ids, vocab_idx, top_pair)
            self.merges[top_pair] = vocab_idx
            merge_count += 1
            vocab_idx += 1
        self._build_vocab()
    
    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)
    
    # Facilitating of Byte-Pair Encoding and utf-8 since in utf-8 after 256th character, every character is represented by
    # multiple bytes so that we can add (concatenate) pair of bytes to represent another unicode code point or a pair of tokens.
    # With that feature of utf-8 we can further decode text via previously created byte stream's corresponding indexes because 
    # every token that the model generally from merged indexes lets say 25894 and this item's corresponding merge tree will be
    # stored as every low level raw character's (0-255) concatenated byte representation which forms byte stream.
    def _build_vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(self.initial_vocab_size)}
        for pair, merged_idx in self.merges.items():
            first, sec = pair
            self.vocab[merged_idx] = self.vocab[first] + self.vocab[sec]
        return self.vocab

    def decode(self, token_sequence):
        byte_stream = b''.join(self.vocab[token_idx] for token_idx in token_sequence)
        return byte_stream.decode('utf-8', errors='replace')

    def encode(self, text):
        return self._encode_chunk(self.byte_encoder(text))
    
    def _encode_chunk(self, tokens):
        # We want to find a pair that has the lowest merge rank because we will contruct our tokens from leaf to root.
        while len(tokens) >= 2:
            stats = self.get_pair_counts(tokens)
            lowest_vocab_index_pair = min(stats, key = lambda pair: self.merges.get(pair, float("inf")))
            if lowest_vocab_index_pair not in self.merges:
                   break
            tokens = self.merge(tokens, self.merges[lowest_vocab_index_pair], lowest_vocab_index_pair)
        return tokens

    def byte_encoder(self, text):
        raw_bytes = text.encode("utf-8") # raw-bytes
        unicode_byte_ids = list(map(int, raw_bytes)) # Convert the raw byte stream into list of integers valued between 0 - 255.
        return unicode_byte_ids

    def _config(self):
        return {"min_frequency": self.min_frequency}

    def save(self, path):
        model = {
            "version": 1,
            "type": type(self).__name__,
            "config": self._config(),
            "initial_vocab_size": self.initial_vocab_size,
            "merges": [[first, sec, idx] for (first, sec), idx in sorted(self.merges.items(), key=lambda item: item[1])],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f)

    def _load_state(self, model):
        self.initial_vocab_size = model["initial_vocab_size"]
        self.merges = {(first, sec): merged_idx for first, sec, merged_idx in model["merges"]}
        self.special_tokens = model["special_tokens"]
        self.vocab = self._build_vocab() 
    
    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
        assert model["type"] == cls.__name__, f"file holds a {model['type']}, not a {cls.__name__}"
        tokenizer = cls(**model["config"])     # same constructor as a brand-new tokenizer
        tokenizer._load_state(model)
        return tokenizer

 