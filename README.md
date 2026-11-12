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
 
## 📥 Datasets Used

Due to GitHub size limits, the full datasets are not included in this repository.

### Job Descriptions Dataset
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/andrewmvd/job-description-dataset
- File used: job_title_des.csv

### Glassdoor Employee Reviews Dataset
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews
- Relevant columns: pros, cons, advice_to_mgmt

Users should download these datasets manually and place them in the project root directory before running the code.


## ✅ How the gender direction is checked

Two accuracies are written to `outputs/metrics.json`:

- `validation_accuracy` scores the same male/female words that were used to
  build the direction. It is circular and will look high on almost any embedding.
- `held_out_accuracy` is the honest one: for each male/female pair the direction
  is rebuilt without that pair, and only the two left-out words are scored
  (leave-one-pair-out). On random vectors with no gender axis this falls to
  about 50% while the circular score stays near 80%.

`metrics.json` also lists `missing_words`: any target, attribute or job word
that does not occur in the corpus and was therefore skipped.

## ⚠️ Limits

- The gender lexicon is 13 word pairs and the WEAT attribute sets have 8 to 9
  words, so effect sizes are indicative, not precise.
- Word2Vec is trained with `workers=4`, so two runs give slightly different vectors.
- Gender is treated as a male/female axis because that is what the lexicon
  encodes; this is a limitation of the method, not a claim about gender.
- The result images below come from a run made **before** a preprocessing bug
  was fixed: the lemmatizer was rewriting `children`, `parents`, `relatives`
  and `sales`, so those words were silently missing from the analysis. The
  pipeline has not been re-run on the full datasets since, so re-run it before
  quoting numbers.

#### 📊 Results
<img width="542" height="86" alt="image" src="https://github.com/user-attachments/assets/40bb95e9-64c9-480c-a0e1-11cc2e0b84af" />
<img width="1440" height="900" alt="image" src="https://github.com/user-attachments/assets/be431b9b-9a94-45e6-bca6-0fc055262fcd" />
<img width="1440" height="900" alt="image" src="https://github.com/user-attachments/assets/6a0c2db0-d0fe-44b9-ac70-dfba16178b75" />
<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/3277ac9c-aa21-4e0c-aec9-afd1e3a33daf" />
<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/9b00515c-5e53-47d8-bad9-69b4a18d5a26" />
<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/6404d6fc-9cdd-4e28-9c08-3c611dbe2862" />

## Running

```bash
pip install -r requirements.txt
python bias_detection.py
```

Results (plots and scores) are written to the `outputs/` folder.

## Tests

```bash
pip install pytest
pytest
```

Nine checks on a hand-built embedding with a known gender axis: WEAT finds a
planted bias, stays quiet without one and flips sign with it; the direction is
male-positive; the held-out check drops to chance on random vectors; and the
text cleaner keeps the exact words the analysis looks up. They need no corpus
and no gensim.
