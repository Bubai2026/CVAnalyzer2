import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os

from model import CVTransformer, build_vocab, encode, CVJDDataset

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/train_pairs.csv")

resumes = df["resume"].tolist()
jobs = df["job"].tolist()
labels = df["label"].tolist()

# ---------------- BUILD VOCAB ----------------
vocab = build_vocab(resumes + jobs)

# ---------------- ENCODE ----------------
resume_enc = [encode(t, vocab) for t in resumes]
job_enc = [encode(t, vocab) for t in jobs]

# ---------------- DATASET ----------------
dataset = CVJDDataset(resume_enc, job_enc, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------------- MODEL ----------------
model = CVTransformer(vocab_size=len(vocab)+1)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ---------------- TRAIN ----------------
for epoch in range(5):
    total_loss = 0

    for r, j, y in loader:
        optimizer.zero_grad()

        output = model(r, j).squeeze()
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ---------------- SAVE ----------------
os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/model.pth")

with open("checkpoints/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
