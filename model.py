import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
from collections import Counter
import pickle

# ---------------- TOKENIZER ----------------

def build_vocab(texts, max_size=5000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word: i+1 for i, (word, _) in enumerate(counter.most_common(max_size))}
    return vocab

def encode(text, vocab, max_len=100):
    tokens = text.split()
    encoded = [vocab.get(t, 0) for t in tokens[:max_len]]
    return encoded + [0] * (max_len - len(encoded))


# ---------------- DATASET ----------------

class CVJDDataset(Dataset):
    def __init__(self, resumes, jobs, labels):
        self.resumes = resumes
        self.jobs = jobs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.resumes[idx]),
            torch.tensor(self.jobs[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


# ---------------- POSITIONAL ENCODING ----------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


# ---------------- TRANSFORMER MODEL ----------------

class CVTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return x.mean(dim=1)

    def forward(self, resume, job):
        r = self.encode(resume)
        j = self.encode(job)
        combined = torch.cat([r, j], dim=1)
        return self.fc(combined)
