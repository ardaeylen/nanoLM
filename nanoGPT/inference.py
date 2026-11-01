import torch
import os
import sys
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Tokenizer")
)

from RegexTokenizer import RegexTokenizer
from utils import load_checkpoint
from model import TransformerModel



if __name__ == "__main__":

    script_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = "/".join(script_path.split("/")[:-1])
    tokenizer_path = os.path.join(project_folder_path, "tokenizer_8k.json")
    model_saving_path = "checkpoints/model.pth"
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
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=dim_model,
        max_context_length=context_length,
        num_heads=head_number,
        num_blocks=num_attn_blocks,
        eot_id=tokenizer_eot_token,
    ).to(device)
    # model = load_checkpoint(model, model_saving_path)
    model_num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameter size: {model_num_parameters}")