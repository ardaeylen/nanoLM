from tokenizer_base import TokenizerBase
import regex as re
from collections import Counter
from typing import Optional, List
from tqdm import tqdm

GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(TokenizerBase):
    def __init__(self, pattern=GPT2_SPLIT_PATTERN, min_frequency: int = 2, special_tokens: Optional[List[str]] = None):
        super().__init__(min_frequency=min_frequency, special_tokens=special_tokens)
        self.pattern = pattern
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size):
        chunk_freqs = Counter(self._split(text))        # " the" stored once, with its count
        chunks = [self.byte_encoder(chunk) for chunk in chunk_freqs]
        freqs = list(chunk_freqs.values())
        self.merges = {}
        for vocab_idx in tqdm(range(self.initial_vocab_size, vocab_size - len(self.special_token_names))):
            stats = {}
            for ids, freq in zip(chunks, freqs):
                self.get_pair_counts(ids, stats, freq) # pairs never cross chunk boundaries
            if not stats: # every chunk is a single token
                break
            top_pair = max(stats, key=stats.get)
            if stats[top_pair] < self.min_frequency:
                break
            chunks = [self.merge(ids, vocab_idx, top_pair) for ids in chunks]
            self.merges[top_pair] = vocab_idx
        
        self._finish_training()

    def encode(self, text):
        ids = []
        for chunk in self._split(text):
            ids.extend(self._encode_chunk(self.byte_encoder(chunk)))
        return ids
    
    def _split(self, text):
        return self.compiled_pattern.findall(text)
    
    def _config(self):
        return {**super()._config(), "pattern": self.pattern}
    
    