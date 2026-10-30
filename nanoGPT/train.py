from utils import load_txt_dataset
import torch
from model import BigramLanguageModel, BagOfWordsLanguageModel, TransformerModel
from tqdm import tqdm
torch.manual_seed(42)

def get_batch(split, train_set, validation_set, context_length, batch_size):
        data = train_set if split == "train" else validation_set
        starting_idxes_batch = torch.randint(len(data) - context_length, (batch_size, ))
        x = torch.stack([torch.tensor(data[starting_idx:starting_idx + context_length], dtype=torch.long) for starting_idx in starting_idxes_batch])
        y = torch.stack([torch.tensor(data[starting_idx + 1 : starting_idx + context_length + 1], dtype=torch.long) for starting_idx in starting_idxes_batch])
        return x, y

def train(model, optimizer, epochs, train_set, validation_set, context_length, batch_size, device):
    
    for epoch in tqdm(range(epochs)):
        if epoch % 100 == 0:
            losses = evaluate_both_sets(model, 10, train_set, validation_set, context_length, batch_size, device)
            print(f"Train loss {losses['train']} Test loss: {losses['test']}")
        x, y = get_batch("train", train_set, validation_set, context_length, batch_size)
        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_both_sets(model, iteration_per_split, train_set, validation_set, context_length, batch_size, device):
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(iteration_per_split)
        for iter in range(iteration_per_split):
            x, y = get_batch(split, train_set, validation_set, context_length, batch_size)
            x = x.to(device)
            y = y.to(device)
            _, loss =  model(x, y)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



if __name__ == "__main__":
    dataset_path = "/Users/arda/Desktop/Projects/LLM/NanoGPTFromScratch/Tokenizer/tiny_shakespeare.txt"
    dataset = load_txt_dataset(dataset_path)

    chars = sorted(list(set(dataset)))
    vocab_size = len(chars)

    stoi = {ch:idx for idx, ch in enumerate(chars)}
    itos = {idx:ch for idx, ch in enumerate(chars)}

    encode = lambda text : [stoi[char] for char in text]
    decode = lambda token_ids : ''.join(itos[idx] for idx in token_ids)

    data = torch.tensor(encode("Hello world"), dtype=torch.long)

    train_ratio = 0.9
    train_set = encode(dataset[:int(train_ratio * len(dataset))])
    validation_set = encode(dataset[int(train_ratio * len(dataset)):])
    context_length = 128 #256
    batch_size = 32 #64
    dim_model = 512
    head_number = 16 #8
    num_attn_blocks = 2 #6
    device = "mps"

    
    # model = BigramLanguageModel(vocab_size=vocab_size, max_context_length=context_length)
    # model = BagOfWordsLanguageModel(vocab_size=vocab_size, hidden_dim=dim_model, max_context_length = context_length)
    model = TransformerModel(vocab_size=vocab_size, hidden_dim=dim_model, max_context_length = context_length, num_heads = head_number, num_blocks=num_attn_blocks).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train(model, optimizer, 1000, train_set, validation_set, context_length, batch_size, device)
    #print(loss)
    model.eval()
    print(decode(model.generate(torch.tensor(encode("hello")).unsqueeze(0).to(device), 1000)[0].tolist()))
