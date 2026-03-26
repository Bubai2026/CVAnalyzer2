import torch
import re
import pickle
from model import CVTransformer, encode

# ---------------- PREPROCESS ----------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------- SKILL EXTRACTION ----------------

def extract_skills(text, skills_db):
    return list(set([s for s in skills_db if s in text]))


# ---------------- LOAD MODEL ----------------

with open("checkpoints/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = CVTransformer(vocab_size=len(vocab)+1)
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()


# ---------------- INPUT ----------------

resume_text = open("resume.txt").read()
job_text = open("job.txt").read()

skills_db = ["python", "c++", "sql", "machine learning", "deep learning", "aws"]

# ---------------- PROCESS ----------------

resume_clean = clean_text(resume_text)
job_clean = clean_text(job_text)

resume_enc = encode(resume_clean, vocab)
job_enc = encode(job_clean, vocab)

# ---------------- PREDICT ----------------

with torch.no_grad():
    score = model(
        torch.tensor([resume_enc]),
        torch.tensor([job_enc])
    ).item()

# ---------------- ANALYSIS ----------------

resume_skills = extract_skills(resume_clean, skills_db)
job_skills = extract_skills(job_clean, skills_db)

matched = set(resume_skills) & set(job_skills)
missing = set(job_skills) - set(resume_skills)

print("Match Score:", round(score * 100, 2))
print("Matched Skills:", matched)
print("Missing Skills:", missing)
