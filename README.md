# 🔍 Gender Bias Detection in Text Using Word Embeddings

This project detects and quantifies **gender bias in job-related language** using **Word2Vec embeddings**, combining **unsupervised anchoring**, **supervised learning**, and **WEAT (Word Embedding Association Test)**.  
It also provides **clear visualizations** showing how different job roles align along a gender direction.

---

## 📌 Project Motivation

Word embeddings learn from real-world text and often capture **implicit societal biases**.  
This project investigates:

- How **job titles and roles** are associated with gender
- Whether embeddings encode **male/female stereotypes**
- How bias can be **measured, validated, and visualized**

---

## 🧠 Project Pipeline (Phases 1–4)

### **Phase 1: Data Preprocessing**
- Tokenization, lemmatization, and stopword removal
- Gender-related and bias-relevant words are **explicitly preserved**
- Data sources:
  - Job descriptions (`job_title_des.csv`)
  - Employee reviews (`glassdoor_reviews.csv`)

---

### **Phase 2: Word Embedding Training**
- Model: **Word2Vec (Skip-Gram)**
- Key parameters:
  - `vector_size = 300`
  - `window = 8`
  - `epochs = 30`
  - `negative sampling = 20`
- Trained on combined corpus for richer semantics

---

### **Phase 3: Bias Measurement**
- **WEAT (Word Embedding Association Test)**
- **Cohen’s d** effect size
- **Permutation test** for significance
- Gender direction constructed using:
  - Anchor word pairs (he–she, man–woman, etc.)
  - Supervised Logistic Regression
- Direction oriented so:
  - **Male → positive**
  - **Female → negative**

#### 📊 Results
