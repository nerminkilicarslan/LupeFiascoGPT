import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, text, block_size=256):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data   = data[n:]

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, l):
        return ''.join([self.itos.get(i, '') for i in l])

    def __len__(self):
        return len(self.train_data) - self.block_size

    def __getitem__(self, idx):
        x = self.train_data[idx:idx + self.block_size]
        y = self.train_data[idx + 1:idx + self.block_size + 1]
        return x, y
