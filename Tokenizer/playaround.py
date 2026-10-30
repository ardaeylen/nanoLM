import copy
def get_pair_counts(ids):
    stats = {}
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats

def merge(id_sequence, new_idx, pair):
    new_id_sequence = []
    num_ids = len(id_sequence)
    idx = 0
    print(f"Initial length: {len(id_sequence)}")
    while idx < num_ids:
        if idx <  num_ids - 1 and id_sequence[idx] == pair[0] and id_sequence[idx + 1] == pair[1]:
            new_id_sequence.append(new_idx)
            idx += 2
        else:
            new_id_sequence.append(id_sequence[idx])
            idx += 1
    print(f"New length: {(len(new_id_sequence))}")
    return new_id_sequence


def train_tokenizer(ids, merges, num_merges, init_vocab_size):
    vocab_idx = init_vocab_size
    merge_count = 0
    while merge_count < num_merges:
        current_stats = get_pair_counts(ids)
        top_pair = max(current_stats, key=current_stats.get)
        ids = merge(ids, vocab_idx, top_pair)
        merges[top_pair] = vocab_idx
        merge_count += 1
        vocab_idx += 1
    return ids

text = """As you can see in CUDA, the key steps after writing our kernel that we need to do are allocating our host and device memory, copying the data from host to device, launching the kernel and copying the results back to host. While Metal avoids us having to copy data back and forth due to the unified memory architecture, we have to do a lot more work to get things set up before we call our kernel.
Calling a Metal kernel (after writing the GPU function in Metal Shading Language) involves creating a compute pipeline from the compiled function, allocating device-accessible buffers for input and output data, constructing a command buffer to hold instructions, encoding the kernel dispatch call into that command buffer, and finally submitting the command buffer to the GPU for execution.
As noted in another blog, I am unsure why some of this boilerplate could not be wrapped into an easy kernel dispatcher function, similar to how CUDA does it. I suspect it is because Apple wants to keep the Metal API as low-level as possible, and give developers more control over the GPU resources. As a beginner to GPU programming, I had to spend a little bit longer trying to learn about command queues, command buffers, data buffers, and pipeline. But they let you have really close control over your CPU and GPU compute cycles and memory, and for someone interested in further understanding how that can be useful in writing performant code can check out this video from Apple: Optimize Metal performance for Apple Silicon Macs.
Parallels in the programming APIs
Similar to how we can map a lot of the concepts from NVIDIA GPU system architecture to Apple Silicon, a lot of the programming constructs in CUDA and Metal have parallels. While I have tried my best to be correct in writing the translation from Metal to CUDA programming contructs, I did use Claude's helo to write the table, so please do let me know if you find any errors."""

tokens = text.encode('utf-8') # raw-bytes
tokens = list(map(int, tokens)) # Convert the raw byte stream into list of integers valued between 0 - 255.
ids = copy.deepcopy(tokens) # Copy the initial token indexes so we don't lose the original text ids.
merges = {} # int -> int
initial_vocab_size = 256
num_merges = 5

ids = train_tokenizer(ids, merges, num_merges, initial_vocab_size)

# Enjoy facilitating of Byte-Pair Encoding and utf-8 Because in utf-8 after 256th character every character is represented by
# multiple bytes so that we can add (concatenate) pair of bytes to represent another unicode code point or a pair of tokens.
# With that feature of utf-8 we can further decode text via previously created byte stream's corresponding indexes because 
# every token that the model generally from merged indexes lets say 25894 and this item's corresponding merge tree will be
# stored as every low level raw character's (0-255) concatenated byte representation which forms byte stream.
def prepare_vocab(merges, initial_vocab_size):
    vocab = {idx: bytes([idx]) for idx in range(initial_vocab_size)}
    for pair, merged_idx in merges.items():
        first, sec = pair
        vocab[merged_idx] = vocab[first] + vocab[sec]
    return vocab

vocab = prepare_vocab(merges, initial_vocab_size)

def decode(token_sequence, vocab):
    byte_stream = b''.join(vocab[token_idx] for token_idx in token_sequence)
    return byte_stream.decode('utf-8', errors='replace')

#print(decode([128], vocab))

def encode(text, merges):
    tokens = text.encode('utf-8') # raw-bytes
    tokens = list(map(int, tokens)) # Convert the raw byte stream into list of integers valued between 0 - 255.
    if len(tokens) <= 1:
        return tokens
    stats = get_pair_counts(tokens)
    # We want to find a pair that has the lowest index in vocabulary because we will contruct our tokens from leaf to root.
    lowest_vocab_index_pair = min(stats, key = lambda pair: merges.get(pair, float("inf"))) 
    while merges.get(lowest_vocab_index_pair, None) in vocab:
        print(f"{lowest_vocab_index_pair} found and will be merged to form an id of {merges.get(lowest_vocab_index_pair)}")
        tokens = merge(tokens, merges[lowest_vocab_index_pair], lowest_vocab_index_pair)
        stats = get_pair_counts(tokens)
        lowest_vocab_index_pair = min(stats, key = lambda pair: merges.get(pair, float("inf"))) 
    return tokens

#print(merges)
text1 = """into an easy kernel dispatcher function, 
similar to how CUDA does it. I suspect it is because Apple wants to keep the Metal API as low-level as possible, and give developers more control over the GPU resources. As a beginner to GPU programming, I had to spend a little bit longer trying to learn about command queues, command buffers, data buffers, and pipeline"""
text2 = decode(encode(text1, merges), vocab) 
print(text1 == text2)
#print(decode(encode("hello world", merges), vocab))

#self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
