import torch
import os
from dataset import CharDataset
from model import GPTLanguageModel

# ---------- Hyperparameters ----------
batch_size    = 32
block_size    = 256
max_iters     = 10000
eval_interval = 500
eval_iters    = 200
learning_rate = 3e-4
n_embd        = 256
n_head        = 8
n_layer       = 6
dropout       = 0.2
# -------------------------------------

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"Cihaz: {device}")
torch.manual_seed(1337)

with open("lupe_dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

dataset = CharDataset(text, block_size)
print(f"Dataset: {len(text):,} karakter  |  Vocab boyutu: {dataset.vocab_size}")


def get_batch(split):
    data = dataset.train_data if split == 'train' else dataset.val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i + block_size]     for i in ix])
    y    = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTLanguageModel(
    vocab_size=dataset.vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
).to(device)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model parametreleri: {total_params:.2f}M\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
os.makedirs("checkpoints", exist_ok=True)

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': dataset.vocab_size,
            'stoi':       dataset.stoi,
            'itos':       dataset.itos,
            'block_size': block_size,
            'n_embd':     n_embd,
            'n_head':     n_head,
            'n_layer':    n_layer,
        }, "checkpoints/model.pt")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nEgitim tamamlandi! -> checkpoints/model.pt")
