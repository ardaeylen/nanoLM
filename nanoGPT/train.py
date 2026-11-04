import torch
from model import BigramLanguageModel, BagOfWordsLanguageModel, CausalTransformer
from tqdm import tqdm
import sys
import os
import numpy as np
from utils import save_checkpoint
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Tokenizer")
)

from RegexTokenizer import RegexTokenizer

torch.manual_seed(42)


def get_batch(split, train_set, validation_set, context_length, batch_size):
    data = train_set if split == "train" else validation_set
    starting_idxes_batch = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack(
        [
            torch.tensor(
                data[starting_idx : starting_idx + context_length], dtype=torch.long
            )
            for starting_idx in starting_idxes_batch
        ]
    )
    y = torch.stack(
        [
            torch.tensor(
                data[starting_idx + 1 : starting_idx + context_length + 1],
                dtype=torch.long,
            )
            for starting_idx in starting_idxes_batch
        ]
    )
    return x, y


def train(
    model,
    optimizer,
    epochs,
    train_set,
    validation_set,
    context_length,
    batch_size,
    device,
    model_saving_path
):
    min_valid_loss = float("inf")
    for epoch in tqdm(range(epochs)):
        
        if epoch % 100 == 0:
            losses = evaluate_both_sets(
                model, 10, train_set, validation_set, context_length, batch_size, device
            )
            print(f"Train loss {losses['train']} Test loss: {losses['test']}")
            if losses["test"] < min_valid_loss:
                save_checkpoint(model, model_saving_path)
                min_valid_loss = losses["test"]
        x, y = get_batch("train", train_set, validation_set, context_length, batch_size)
        x = x.to(device)
        y = y.to(device)

        _, loss, _ = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_both_sets(
    model,
    iteration_per_split,
    train_set,
    validation_set,
    context_length,
    batch_size,
    device,
):
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(iteration_per_split)
        for iter in range(iteration_per_split):
            x, y = get_batch(
                split, train_set, validation_set, context_length, batch_size
            )
            x = x.to(device)
            y = y.to(device)
            _, loss, _ = model(x, y)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = "/".join(script_path.split("/")[:-1])
    tokenizer_path = os.path.join(project_folder_path, "tokenizer_8k.json")
   
    model_saving_path = os.path.join(script_path,"checkpoints/model.pth")
    tokenizer = RegexTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    tokenizer_eot_token = tokenizer.special_tokens["<|endoftext|>"]
    data = np.memmap(
        os.path.join(project_folder_path, "data/tinystories_8k.bin"),
        dtype=np.uint16,
        mode="r",
    )
    length_data = len(data)
    train_ratio = 0.99

    train_set = data[: int(train_ratio * length_data)]
    validation_set = data[int(train_ratio * length_data) :]

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
        vocab_size=vocab_size,
        hidden_dim=dim_model,
        max_context_length=context_length,
        num_heads=head_number,
        num_blocks=num_attn_blocks,
        eot_id=tokenizer_eot_token,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train(
        model,
        optimizer,
        num_epochs,
        train_set,
        validation_set,
        context_length,
        batch_size,
        device,
        model_saving_path,
    )

    model.eval()
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
