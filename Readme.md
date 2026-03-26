# 📄 CV Analyzer using Transformer (From Scratch)

## 🚀 Overview

This project is a **CV Analyzer** that evaluates how well a resume matches a given job description using **Natural Language Processing (NLP)** and a **custom Transformer-based deep learning model built from scratch**.

Unlike basic keyword matching systems, this project uses a **Siamese Transformer Encoder architecture** to capture semantic relationships between resumes and job descriptions.

---

## 🎯 Objectives

* Parse and preprocess resume and job description text
* Remove noise and irrelevant words (stopwords, symbols)
* Learn semantic similarity using a custom Transformer model
* Predict a **match score (0–100%)**
* Identify:

  * ✅ Matching skills
  * ❌ Missing skills

---

## 🧠 Model Architecture

This project implements a **Transformer Encoder from scratch** using PyTorch.

### 🔹 Architecture Flow

```
Resume → Transformer Encoder → Embedding
Job    → Transformer Encoder → Embedding

→ Concatenate → Dense Layers → Match Score
```

### 🔹 Key Components

* Word Embedding Layer
* Positional Encoding
* Multi-head Self-Attention
* Feedforward Layers
* Mean Pooling
* Fully Connected Layers (for similarity scoring)

---

## ⚙️ Pipeline

### 🔹 Step 1: Input

* Resume (text file)
* Job Description (text file)

### 🔹 Step 2: Preprocessing

* Lowercasing
* Removing special characters
* Removing extra spaces

### 🔹 Step 3: Tokenization

* Build vocabulary
* Convert text → numerical sequences

### 🔹 Step 4: Model Inference

* Encode resume & job description
* Compute similarity score using Transformer

### 🔹 Step 5: Analysis

* Extract skills
* Compare resume vs job skills
* Output:

  * Match score
  * Matching skills
  * Missing skills

---

## 📊 Dataset

This project uses a **custom paired dataset**:

```
(resume_text, job_description, label)
```

* `label = 1` → good match
* `label = 0` → poor match

### 🔹 Dataset Format (`train_pairs.csv`)

| resume | job  | label |
| ------ | ---- | ----- |
| text   | text | 1/0   |

---

## 📁 Project Structure

```
CV_Analyzer/
│
├── main.py        # Full pipeline (parse + preprocess + predict)
├── train.py       # Training script
├── model.py       # Transformer model + tokenizer + dataset
│
├── data/
│   └── train_pairs.csv
│
├── checkpoints/
│   ├── model.pth
│   └── vocab.pkl
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/CV_Analyzer.git
cd CV_Analyzer
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Step 1: Train the Model

```bash
python train.py
```

This will:

* Train the Transformer model
* Save:

  * `model.pth`
  * `vocab.pkl`

---

### 🔹 Step 2: Run Analysis

Place:

* `resume.txt`
* `job.txt`

Then run:

```bash
python main.py
```

---

## 📈 Sample Output

```
Match Score: 78.45

Matched Skills: {'python', 'machine learning'}
Missing Skills: {'aws', 'docker'}
```

---

## 🧩 Key Features

* ✅ Transformer model built from scratch (no pre-trained models)
* ✅ End-to-end ML pipeline
* ✅ Semantic matching (not just keyword matching)
* ✅ Skill gap analysis
* ✅ Minimal and clean architecture

---

## ⚠️ Limitations

* Dataset is not pre-aligned → requires manual pairing
* No pretraining → performance depends on dataset size
* Basic skill extraction (rule-based)

---

## 🚀 Future Improvements

* Replace basic tokenizer with subword tokenization
* Add Named Entity Recognition (NER)
* Use contrastive / triplet loss instead of BCE
* Add UI using Streamlit
* Rank multiple resumes
* Visualize attention weights

---

## 🧠 Learning Outcomes

This project demonstrates:

* Deep understanding of:

  * Transformer architecture
  * NLP preprocessing
  * Model training pipelines
* Ability to:

  * Build ML systems from scratch
  * Design modular architecture
  * Solve real-world problems

---

## 💬 How to Explain in Interview

> "I built a CV Analyzer using a custom Transformer Encoder to measure semantic similarity between resumes and job descriptions. The model learns to predict match scores and identify skill gaps, enabling more intelligent candidate evaluation than traditional keyword-based systems."

---

## 📌 Requirements

Example:

```
torch
pandas
numpy
scikit-learn
```

---

## ⭐ Final Note

This project focuses on **understanding and building core ML concepts from scratch**, rather than relying on pre-trained models — making it highly valuable for interviews and deep learning fundamentals.
