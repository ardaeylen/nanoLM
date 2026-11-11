import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_context_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_context_len = max_context_length

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.positional_encoding = nn.Embedding(self.max_context_len, self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, batch_id_sequence, target_batch_id_sequence):
        B, T = batch_id_sequence.shape

        # input and targets have both shape (B, T)
        tok_embs = self.token_embedding_table(batch_id_sequence)
        # Positional embedding tensor is of shape (B, T)
        pos_embs = self.positional_encoding(
            torch.arange(0, T, device=batch_id_sequence.device)
        )
        # (B, T, C) + (T, C) -> (B, T, C)
        x = tok_embs + pos_embs
        # Now that logits is of shape (B, T, C)
        logits = self.lm_head(x)
        # Now the logits have shape (B, T, vocab_size)

        if target_batch_id_sequence is None:
            return logits, None
        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = target_batch_id_sequence.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, input_ids, max_new_tokens):

        for _ in range(max_new_tokens):
            # logits is of shape (B, T, C)
            logits, _ = self(input_ids[:, -self.max_context_len :], None)
            # logits -> (B, C)
            logits = logits[:, -1, :]
            # probabilities -> (B, C)
            probabilities = F.softmax(logits, dim=-1)
            # idx_next -> (B,1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            # input_ids -> (B, T + 1)
            input_ids = torch.concat([input_ids, idx_next], dim=-1)

        return input_ids


class BagOfWordsLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_context_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_context_len = max_context_length
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.positional_encoding = nn.Embedding(self.max_context_len, self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, batch_id_sequence, target_batch_id_sequence):
        B, T = batch_id_sequence.shape
        # input and targets have both shape (B, T)
        tok_embs = self.token_embedding_table(batch_id_sequence)
        # Positional embedding tensor is of shape (B, T)
        pos_embs = self.positional_encoding(
            torch.arange(0, T, device=batch_id_sequence.device)
        )
        # (B, T, C) + (T, C) -> (B, T, C)
        x = tok_embs + pos_embs
        # Now that logits is of shape (B, T, C)
        logits = self.lm_head(x)
        # Now the logits have shape (B, T, vocab_size)
        if target_batch_id_sequence is None:
            return logits, None
        else:
            B, T, C = logits.shape
            # Tril mask is used for mean operation across sequence from the beginning.
            # [1, 0] @ [2, 3] -> [2, 3]
            # [1, 1]   [5, 6] -> [7, 9] it's basically a cumulative sum when we imagine that sequence is arranged
            # row ordered where colums are different channels. trill_mask @ (T, C) vector sums the input in row dim.
            mean_mask = torch.tril(
                torch.ones(T, T)
            ).device(batch_id_sequence.device)  # Now that mean mask has shape (T, T) but with this values it only sums.
            mean_mask = mean_mask / torch.sum(
                mean_mask, dim=1, keepdim=True
            )  # Now that our mean multiplicator matrix is ready.
            # [1.0, 0.0] is now mean mask.
            # [0.5, 0.5]

            # =====================================================================================================
            # The other way of taking a mean across sequence dimension.
            # lower_triangular_ones = torch.tril(torch.ones(T, T))
            # actual_mask = torch.zeros((T, T), dtype=logits.dtype)
            # actual_mask = actual_mask.masked_fill(lower_triangular_ones == 0, float("-inf"))
            # actual_mask = F.softmax(actual_mask, dim=1) # Since softmax calculates e^logit and the logit is -inf
            # in upper triangular part, the upper triangle becomes 0, and the other 0 values in mask become the average.
            # e.g. [1, 1, -inf] -> [1/2, 1/2, 0]
            # logits2 = actual_mask @ logits
            # =====================================================================================================

            logits = mean_mask @ logits
            # print(torch.allclose(logits2, logits))

            logits = logits.view(B * T, C)
            targets = target_batch_id_sequence.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, input_ids, max_new_tokens):

        for _ in range(max_new_tokens):
            # logits is of shape (B, T, C)
            logits, _ = self(input_ids[:, -self.max_context_len :], None)
            # logits -> (B, C)
            B, T, C = logits.shape
            mean_mask = torch.tril(torch.ones(T, T))
            mean_mask = mean_mask / torch.sum(mean_mask, dim=1, keepdim=True)
            logits = mean_mask @ logits
            logits = logits[:, -1, :]
            # probabilities -> (B, C)
            probabilities = F.softmax(logits, dim=-1)
            # idx_next -> (B,1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            # input_ids -> (B, T + 1)
            input_ids = torch.concat([input_ids, idx_next], dim=-1)

        return input_ids


class AttentionBlock(nn.Module):
    def __init__(self, model_dim, head_dim, query_key_dim, value_dim, context_length):
        super().__init__()
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.query_dim = query_key_dim
        self.key_dim = query_key_dim
        self.value_dim = value_dim
        self.context_length = context_length

        # Tril tensor is not a variable of the module instead according to the naming conventions of pytorch
        # it's a buffer.
        self.register_buffer(
            "tril_tensor",
            torch.tril(
                torch.ones((self.context_length, self.context_length)).view(
                    1, self.context_length, self.context_length
                )
            ),
        )
        self.query_weigths = nn.Linear(
            in_features=self.model_dim, out_features=self.query_dim, bias=False
        )
        self.key_weights = nn.Linear(
            in_features=self.model_dim, out_features=self.key_dim, bias=False
        )
        self.value_weights = nn.Linear(
            in_features=self.model_dim, out_features=self.value_dim, bias=False
        )
        self.fully_connected = nn.Linear(
            in_features=self.value_dim, out_features=self.head_dim
        )

    def forward(self, sequence_features):
        B, T, C = sequence_features.shape
        # Input sequence is of shape (B, T, C)
        queries = self.query_weigths(sequence_features)  # q -> (B, T, q_head_dim)
        keys = self.key_weights(sequence_features)  # k -> (B, T, kv_head_dim)
        values = self.value_weights(sequence_features)  # v -> (B, T, kv_head_dim)
        # To scale the attention weights in order to control variance of the dot product.
        # Because even if multiplied matrices both came from a gaussian distribution
        # Their dot product's variance may not be gaussian at all with greater variance.
        scale_factor = self.query_dim**0.5

        # (B, T, C) @ (B, C, T) -> (B, T, T)
        attention_weigths = queries @ keys.transpose(2, 1) / scale_factor
        # self. tril is a (T_max x T_max) lower triangular mask.
        # Attention scores' upper right triangle is filled with -inf to zero out in softmax.
        # Masked weights is also (B, T, T)
        masked_weights = attention_weigths.masked_fill(
            self.tril_tensor[:, :T, :T] == 0, float("-inf")
        )
        softmax_weights = F.softmax(masked_weights, dim=-1)

        # In order to extract relational scores we dot product values with attention weights
        # to get weighted sum of token features where weights are analogically relations between them
        # Therefore (B, T, T) @ (B, T, v_head_dim) -> (B, T, v_head_dim)
        features = softmax_weights @ values

        # In order to position-wise transform every token feature and stack multiple blocks one after another
        # at the end of attention operation, we pass the features to fully connected layer to get (B, T, C) again.
        features = self.fully_connected(features)
        return features


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, query_key_dim, value_dim, context_length):
        super().__init__()
        self.model_embed_dim = model_dim
        self.num_heads = num_heads
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.max_context_len = context_length

        self.head_size = self.model_embed_dim // self.num_heads

        assert (self.head_size * self.num_heads) == self.model_embed_dim
        self.scale_factor = self.head_size**0.5

        self.register_buffer(
            "tril_tensor",
            torch.tril(
                torch.ones((self.max_context_len, self.max_context_len)).view(
                    1, 1, self.max_context_len, self.max_context_len
                )
            ),
        )

        self.query_weights = nn.Linear(
            in_features=self.model_embed_dim,
            out_features=self.model_embed_dim,
            bias=False,
        )
        self.key_weights = nn.Linear(
            in_features=self.model_embed_dim,
            out_features=self.model_embed_dim,
            bias=False,
        )
        self.value_weights = nn.Linear(
            in_features=self.model_embed_dim,
            out_features=self.model_embed_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            in_features=self.model_embed_dim, out_features=self.model_embed_dim
        )
        self.drop_out = nn.Dropout(0.2)

    def forward(self, input_sequence, kv_cache = None):
        B, T, C = input_sequence.shape
        # Just a linear transformation for input sequence of shape (B, T, C) to (B, T, C) for each attention weights q, k, v.
        queries = self.query_weights(input_sequence)
        keys = self.key_weights(input_sequence)
        values = self.value_weights(input_sequence)

        # Divide the q, k, v matrices into number of heads to form a shape of (B, num_heads, T, head_dim)

        queries = queries.view(B, T, self.num_heads, self.head_size)
        keys = keys.view(B, T, self.num_heads, self.head_size)
        values = values.view(B, T, self.num_heads, self.head_size)
        # Now that queries, keys and values are all (B, T, #heads, head_dim) dimension tensors.

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # Now that queries, keys and values are all (B, #heads, T, head_dim) dimension tensors.

        # KV Cache
        if kv_cache is not None:
            # Concatenating (B, #heads, T_past, head_dim) of past keys, values with (B, #heads, T_new, head_dim) new keys values on sequence dim.
            # In the inference time, batch of latest generated elements will be passed. Therefore no need to do anything for queries.
            keys_past, values_past = kv_cache

            keys = torch.cat([keys_past, keys], dim=2) 
            values = torch.cat([values_past, values], dim=2) 
        new_cache = (keys, values)

        T_total = keys.shape[2]
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T).
        attention_scores = queries @ keys.transpose(2, 3) / self.scale_factor
        
        # Fill upper triangular values with -inf to avoid next tokens to participate in softmax.
        masked_attention_weights = attention_scores.masked_fill(
            self.tril_tensor[:, :, T_total - T: T_total, : T_total] == 0, float("-inf")
        )

        masked_attention_weights = F.softmax(masked_attention_weights, dim=-1)
        # Dot product of attention weights with value vectors. So the output feature tensor is of shape (B, num_heads, T, head_dim)

        # Just for regularization of the model in training we drop out some affinities between tokens.
        masked_attention_weights = self.drop_out(masked_attention_weights)

        seq_features = masked_attention_weights @ values
        # Rearrange output features to feed into projection layer. (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) to form (B, T, C = num_heads * head_dim)
        # seq_features -> (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim)
        seq_features = seq_features.transpose(1, 2).contiguous()
        # View only change how data read from memory in case of strides. In contrast, transposing a tensor changes ordering of data
        # so the data needs to be sit on a contiguous memory to redefine its strides and ordering.

        seq_features = seq_features.view(B, T, C)

        projected_features = self.out_proj(seq_features)

        # Just for regularization of training
        projected_features = self.drop_out(projected_features)

        return projected_features, new_cache


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, model_dim, num_heads, query_key_dim, value_dim, context_length):
        super().__init__()
        self.model_embed_dim = model_dim
        self.num_heads = num_heads
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.max_context_len = context_length

        self.head_size = self.model_embed_dim // self.num_heads

        assert (self.head_size * self.num_heads) == self.model_embed_dim
        self.scale_factor = self.head_size**0.5

        self.register_buffer(
            "tril_tensor",
            torch.tril(
                torch.ones((self.max_context_len, self.max_context_len)).view(
                    1, 1, self.max_context_len, self.max_context_len
                )
            ),
        )
        
        self.query_weights = nn.Linear(
            in_features=self.model_embed_dim,
            out_features=self.model_embed_dim,
            bias=False,
        )
        self.key_weights = nn.Linear(
            in_features=self.model_embed_dim,
            out_features=self.model_embed_dim,
            bias=False,
        )
        self.value_weights = nn.Linear(
            in_features=self.model_embed_dim,
            out_features=self.model_embed_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            in_features=self.model_embed_dim, out_features=self.model_embed_dim
        )
        self.drop_out = nn.Dropout(0.2)

    def forward(self, input_sequence, cos_angles, sin_angles, kv_cache = None):
        B, T, C = input_sequence.shape
        # Just a linear transformation for input sequence of shape (B, T, C) to (B, T, C) for each attention weights q, k, v.
        queries = self.query_weights(input_sequence)
        keys = self.key_weights(input_sequence)
        values = self.value_weights(input_sequence)

        # Divide the q, k, v matrices into number of heads to form a shape of (B, num_heads, T, head_dim)

        queries = queries.view(B, T, self.num_heads, self.head_size)
        keys = keys.view(B, T, self.num_heads, self.head_size)
        values = values.view(B, T, self.num_heads, self.head_size)
        # Now that queries, keys and values are all (B, T, #heads, head_dim) dimension tensors.

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # Now that queries, keys and values are all (B, #heads, T, head_dim) dimension tensors.

        # RoPE        
        queries = self.apply_rope(queries, cos_angles, sin_angles)
        keys = self.apply_rope(keys, cos_angles, sin_angles)

        # KV Cache
        if kv_cache is not None:
            # Concatenating (B, #heads, T_past, head_dim) of past keys, values with (B, #heads, T_new, head_dim) new keys values on sequence dim.
            # In the inference time, batch of latest generated elements will be passed. Therefore no need to do anything for queries.
            keys_past, values_past = kv_cache

            keys = torch.cat([keys_past, keys], dim=2) 
            values = torch.cat([values_past, values], dim=2) 
        new_cache = (keys, values)

        T_total = keys.shape[2]
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T).
        attention_scores = queries @ keys.transpose(2, 3) / self.scale_factor
        
        # Fill upper triangular values with -inf to avoid next tokens to participate in softmax.
        masked_attention_weights = attention_scores.masked_fill(
            self.tril_tensor[:, :, T_total - T: T_total, : T_total] == 0, float("-inf")
        )

        masked_attention_weights = F.softmax(masked_attention_weights, dim=-1)
        # Dot product of attention weights with value vectors. So the output feature tensor is of shape (B, num_heads, T, head_dim)

        # Just for regularization of the model in training we drop out some affinities between tokens.
        masked_attention_weights = self.drop_out(masked_attention_weights)

        seq_features = masked_attention_weights @ values
        # Rearrange output features to feed into projection layer. (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) to form (B, T, C = num_heads * head_dim)
        # seq_features -> (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim)
        seq_features = seq_features.transpose(1, 2).contiguous()
        # View only change how data read from memory in case of strides. In contrast, transposing a tensor changes ordering of data
        # so the data needs to be sit on a contiguous memory to redefine its strides and ordering.

        seq_features = seq_features.view(B, T, C)

        projected_features = self.out_proj(seq_features)

        # Just for regularization of training
        projected_features = self.drop_out(projected_features)

        return projected_features, new_cache
    
    @staticmethod
    def apply_rope(x, cos, sin):
        B, num_heads, T, head_dim = x.shape
        x = x.view(B, num_heads, T, head_dim // 2, 2)
        x = x * cos + torch.flip(x * sin, dims=[-1])
        x = x.view(B, num_heads, T, head_dim)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.project_up = nn.Linear(
            in_features=self.embed_dim, out_features=self.proj_dim
        )
        self.project_down = nn.Linear(
            in_features=self.proj_dim, out_features=self.embed_dim
        )
        self.drop_out = nn.Dropout(0.2)

    def forward(self, input_sequence):
        x = F.relu(self.project_up(input_sequence))
        x = self.drop_out(self.project_down(x))
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, model_dim, num_heads, query_key_dim, value_dim, context_length):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.qk_dim = query_key_dim
        self.value_dim = value_dim
        self.context_len = context_length

        self.multi_head_attention = MultiHeadAttention(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            query_key_dim=self.qk_dim,
            value_dim=self.value_dim,
            context_length=self.context_len,
        )
        self.feed_forward = FeedForward(
            embed_dim=self.model_dim, proj_dim=self.model_dim * 4
        )
        # LayerNorm is a normalization across the units of an embedding vectors
        # so that each feature of a token's hidden feature has a gaussian distribution
        # meaning that 0 mean and 1 variance.
        self.layer_norm1 = nn.LayerNorm(self.model_dim)
        self.layer_norm2 = nn.LayerNorm(self.model_dim)
    def forward(self, x, kv_cache = None):
        out, new_cache = self.multi_head_attention(self.layer_norm1(x), kv_cache)
        x = x + out
        x = x + self.feed_forward(self.layer_norm2(x))
        return x, new_cache

class MultiHeadAttentionBlockRoPE(nn.Module):
    def __init__(self, model_dim, num_heads, query_key_dim, value_dim, context_length):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.qk_dim = query_key_dim
        self.value_dim = value_dim
        self.context_len = context_length

        self.multi_head_attention = MultiHeadAttentionRoPE(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            query_key_dim=self.qk_dim,
            value_dim=self.value_dim,
            context_length=self.context_len,
        )
        self.feed_forward = FeedForward(
            embed_dim=self.model_dim, proj_dim=self.model_dim * 4
        )
        # LayerNorm is a normalization across the units of an embedding vectors
        # so that each feature of a token's hidden feature has a gaussian distribution
        # meaning that 0 mean and 1 variance.
        self.layer_norm1 = nn.LayerNorm(self.model_dim)
        self.layer_norm2 = nn.LayerNorm(self.model_dim)
    def forward(self, x, cos_angles, sin_angles, kv_cache = None):
        out, new_cache = self.multi_head_attention(self.layer_norm1(x), cos_angles, sin_angles, kv_cache)
        x = x + out
        x = x + self.feed_forward(self.layer_norm2(x))
        return x, new_cache

class MultiHeadAttentionkarpathy(nn.Module):
    def __init__(self, model_dim, num_heads, query_key_dim, value_dim, context_length):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = self.model_dim // self.num_heads
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.max_context_len = context_length
        assert self.model_dim == int(self.num_heads * self.head_dim)
        self.attention_heads = nn.ModuleList(
            [
                AttentionBlock(
                    self.model_dim,
                    self.head_dim,
                    self.query_key_dim,
                    self.value_dim,
                    self.max_context_len,
                )
                for _ in range(self.num_heads)
            ]
        )

    def forward(self, input_sequence):
        return torch.cat(
            [attention_head(input_sequence) for attention_head in self.attention_heads],
            dim=-1,
        )


class CausalTransformer(nn.Module):
    def __init__(
        self, vocab_size, hidden_dim, max_context_length, num_heads, num_blocks, eot_id
    ):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.eot_id = eot_id
        self.hidden_dim = hidden_dim
        self.max_context_len = max_context_length
        self.head_dim = hidden_dim // num_heads
        self.num_blocks = num_blocks
        assert self.head_dim * num_heads == self.hidden_dim

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.hidden_dim)
        # self.positional_encoding = nn.Embedding(self.max_context_len, self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)

        # Rotary Position Embeddings
        inverse_frequencies = 10000 ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        positions = torch.arange(0, self.max_context_len, dtype=torch.float32)
        freqs = torch.outer(positions, inverse_frequencies)
        sin_angles = torch.sin(freqs).view(1, 1, self.max_context_len, self.head_dim // 2, 1)
        cos_angles = torch.cos(freqs).view(1, 1, self.max_context_len, self.head_dim // 2, 1)
        self.register_buffer("sin_angles", torch.cat([sin_angles, -sin_angles], dim = 4))
        self.register_buffer("cos_angles", torch.cat([cos_angles, cos_angles], dim = 4))



        self.attn_blocks = nn.ModuleList(
            [
                MultiHeadAttentionBlockRoPE(
                    model_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    query_key_dim=self.head_dim,
                    value_dim=self.head_dim,
                    context_length=self.max_context_len,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm_last = nn.LayerNorm(self.hidden_dim)

    def forward(self, batch_id_sequence, target_batch_id_sequence = None, kv_caches = None, start_pos = 0):
        B, T = batch_id_sequence.shape
        # input and targets have both shape (B, T)
        tok_embs = self.token_embedding_table(batch_id_sequence)
        assert start_pos + T <= self.max_context_len, f"position {start_pos + T} > {self.max_context_len}"
        # Positional embedding tensor is of shape (B, T)
        # pos_embs = self.positional_encoding(
        #     torch.arange(start_pos, start_pos + T, device=batch_id_sequence.device)
        # )
        # (B, T, C) + (T, C) -> (B, T, C) (Broadcasted)
        x = tok_embs #+ pos_embs
        cos_angles = self.cos_angles[:, :, start_pos: start_pos + T, :, :]
        sin_angles = self.sin_angles[:, :, start_pos: start_pos + T, :, :]
        new_caches = []
        for block_idx, attn_block in enumerate(self.attn_blocks):
            x, cache = attn_block(x, cos_angles, sin_angles, None if kv_caches is None else kv_caches[block_idx])
            new_caches.append(cache)

        x = self.layer_norm_last(x)
        # Now that input sequence feature is of shape (B, T, C)
        logits = self.lm_head(x)
        # Now the logits have shape (B, T, vocab_size)
        if target_batch_id_sequence is None:
            return logits, None, new_caches
        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = target_batch_id_sequence.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits.cpu(), loss.cpu(), new_caches
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens):
        kv_caches = None
        token_pos = 0
        for _ in range(max_new_tokens):
            # logits is of shape (B, T, C)
            if kv_caches is None:
                logits, _, kv_caches = self(input_ids[:, -self.max_context_len :], None, kv_caches)
            else:
                logits, _, kv_caches = self(input_ids[:, -1 :], None, kv_caches, token_pos)
            # logits -> (B, C)
            B, T, C = logits.shape
            token_pos += T
            if token_pos >= self.max_context_len:
                kv_caches = None
                token_pos = 0
            logits = logits[:, -1, :]
            # probabilities -> (B, C)
            probabilities = F.softmax(logits, dim=-1)
            # idx_next -> (B,1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            # input_ids -> (B, T + 1)
            input_ids = torch.concat([input_ids, idx_next], dim=-1)
            if idx_next.item() == self.eot_id:
                break
        return input_ids


if __name__ == "__main__":
    # block = AttentionBlock(128, 16, 32, 36)

    # rand_int = torch.rand((4, 8, 128))

    # features, masked_weights = block(rand_int)

    # model = MultiHeadAttention(128, 4, 32, 32, 256)
    # rand_int = torch.rand((4, 86, 128))
    # features = model(rand_int)
    # print(features.shape)
    # print(masked_weights.shape)
    pass