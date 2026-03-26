import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import math
from collections import Counter
import pickle
import os

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
            torch.tensor(self.resumes[idx], dtype=torch.long),
            torch.tensor(self.jobs[idx], dtype=torch.long),
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

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
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

        # NOTE: No Sigmoid here (we use BCEWithLogitsLoss)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode(self, x):
        mask = (x == 0)

        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        # -------- Masked Mean Pooling --------
        mask = (~mask).unsqueeze(-1)
        x = x * mask

        summed = x.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        return summed / counts

    def forward(self, resume, job):
        r = self.encode(resume)
        j = self.encode(job)
        combined = torch.cat([r, j], dim=1)
        return self.fc(combined)


# ---------------- LOAD DATA ----------------

df = pd.read_csv("data/train_pairs.csv")

resumes = df["resume"].astype(str).tolist()
jobs = df["job"].astype(str).tolist()
labels = df["label"].tolist()

# ---------------- BUILD VOCAB ----------------

vocab = build_vocab(resumes + jobs)

# ---------------- ENCODE ----------------

resume_enc = [encode(t, vocab) for t in resumes]
job_enc = [encode(t, vocab) for t in jobs]

# ---------------- DATASET ----------------

dataset = CVJDDataset(resume_enc, job_enc, labels)

# Shuffle dataset
dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset)))

# Train/Validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ---------------- MODEL ----------------

model = CVTransformer(vocab_size=len(vocab)+1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

best_val_loss = float("inf")

# ---------------- TRAIN ----------------

for epoch in range(5):
    model.train()
    total_loss = 0

    for r, j, y in train_loader:
        r, j, y = r.to(device), j.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(r, j).squeeze()
        loss = criterion(output, y)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for r, j, y in val_loader:
            r, j, y = r.to(device), j.to(device), y.to(device)

            output = model(r, j).squeeze()
            loss = criterion(output, y)

            val_loss += loss.item()

            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {accuracy:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/model.pth")

# Save vocab
with open("checkpoints/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Training complete. Best model saved!")
