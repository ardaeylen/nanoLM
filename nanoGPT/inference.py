import torch
import os
import sys
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Tokenizer")
)

from RegexTokenizer import RegexTokenizer
from utils import load_checkpoint
from model import CausalTransformer



if __name__ == "__main__":

    script_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = "/".join(script_path.split("/")[:-1])
    tokenizer_path = os.path.join(project_folder_path, "tokenizer_8k.json")
    model_saving_path = os.path.join(script_path,"checkpoints/model.pth")
    tokenizer = RegexTokenizer.load(tokenizer_path)
    tokenizer_eot_token = tokenizer.special_tokens["<|endoftext|>"]

    context_length = 128
    batch_size = 64
    dim_model = 512
    head_number = 16  # 8
    num_attn_blocks = 4  # 6
    num_epochs = 5000
    device = "mps"

    # model = BigramLanguageModel(vocab_size=vocab_size, max_context_length=context_length)
    # model = BagOfWordsLanguageModel(vocab_size=vocab_size, hidden_dim=dim_model, max_context_length = context_length)
    model = CausalTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=dim_model,
        max_context_length=context_length,
        num_heads=head_number,
        num_blocks=num_attn_blocks,
        eot_id=tokenizer_eot_token,
    ).to(device)
    model = load_checkpoint(model, model_saving_path)
    model_num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameter size: {model_num_parameters}")

    model.eval()
    # ids = torch.randint(0, tokenizer.vocab_size, (1, 40), device=device)
    # full_logits, _ , _ = model(ids)                                   # no cache: all 40 at once

    # logits, _, caches = model(ids[:, :10], kv_caches=None, start_pos=0)          # prefill 10
    # assert torch.allclose(logits, full_logits[:, :10], atol=1e-4)
    # for t in range(10, 40):                                                    # decode one by one
    #     logits, _, caches = model(ids[:, t:t+1], kv_caches=caches, start_pos=t)
    #     assert torch.allclose(logits[:, -1], full_logits[:, t], atol=1e-4)

    print(
            tokenizer.decode(
                model.generate(
                    torch.tensor([tokenizer_eot_token])
                    .unsqueeze(0)
                    .to(device),
                    1000,
                )[0].tolist()
            )
        )