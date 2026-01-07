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
<img width="542" height="86" alt="image" src="https://github.com/user-attachments/assets/40bb95e9-64c9-480c-a0e1-11cc2e0b84af" />
<img width="1440" height="900" alt="image" src="https://github.com/user-attachments/assets/be431b9b-9a94-45e6-bca6-0fc055262fcd" />
<img width="1440" height="900" alt="image" src="https://github.com/user-attachments/assets/6a0c2db0-d0fe-44b9-ac70-dfba16178b75" />
<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/3277ac9c-aa21-4e0c-aec9-afd1e3a33daf" />
<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/9b00515c-5e53-47d8-bad9-69b4a18d5a26" />
<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/6404d6fc-9cdd-4e28-9c08-3c611dbe2862" />





