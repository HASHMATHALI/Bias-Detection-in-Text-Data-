"""
Bias Detection in Text Data – Improved (Anchored + Supervised Direction)
Phases 1-4: Preprocessing → Word2Vec → Bias Measurement → Visualization
Male = positive, Female = negative (enforced)
"""


# Imports

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression



# Config

DOC_LIMIT = None           # <<< use full data for best accuracy (set to 20000 for quick tests)
OUT_DIR = Path("outputs")

# Stronger Word2Vec hyperparameters
W2V_PARAMS = dict(
    vector_size=300,
    window=8,
    min_count=1,    # keep pronouns
    workers=4,
    sg=1,           # Skip-Gram
    negative=20,
    epochs=30,
    sample=1e-4     # subsampling of very frequent tokens -> crisper vectors
)

# Bias sets
BIAS_SETS = {
    "male": [
        "he","him","his","himself","man","male","boy","brother","son",
        "husband","father","gentleman","sir"
    ],
    "female": [
        "she","her","hers","herself","woman","female","girl","sister","daughter",
        "wife","mother","lady","madam"
    ],
    "career": ["career","corporation","salary","office","management","professional","promotion","business"],
    "family": ["family","home","children","parents","wedding","marriage","relatives","caring","nurturing"]
}

ANCHOR_PAIRS = [
    ("he","she"), ("him","her"), ("his","hers"),
    ("man","woman"), ("male","female"), ("boy","girl"),
    ("father","mother"), ("brother","sister"),
    ("son","daughter"), ("husband","wife")
]

JOB_WORDS = [
    "engineer","scientist","developer","programmer","analyst",
    "nurse","teacher","receptionist","babysitter","assistant",
    "manager","leader","sales","hr","marketing","accountant",
    "ceo","cto","cfo","intern","director","designer","product","data"
]


# Phase 1 — Preprocessing

def protected_words():
    """Words the analysis looks up by exact spelling, so preprocessing must not alter them."""
    words = set(JOB_WORDS)
    for group in BIAS_SETS.values():
        words.update(group)
    return words


def build_cleaner():
    """Return clean_text(doc) -> tokens. Downloads the NLTK data it needs on first use."""
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        nltk.download(resource, quiet=True)

    base_sw = set(stopwords.words("english"))
    keep = protected_words()
    lem = WordNetLemmatizer()

    def clean_text(doc: str):
        text = str(doc).lower()
        tok = word_tokenize(text)
        tok = [t for t in tok if t.isalpha()]
        # remove stopwords EXCEPT the words we want to keep for bias axis
        tok = [t for t in tok if (t not in base_sw) or (t in keep)]
        # Never lemmatize a protected word: the lemmatizer turns "children" into
        # "child", "parents" into "parent" and "sales" into "sale", which silently
        # removed those words from the WEAT sets and the job list.
        tok = [t if t in keep else lem.lemmatize(t) for t in tok]
        return [w for w in tok if len(w) > 1]

    return clean_text


def phase1_preprocessing():
    print("\n Phase 1: Data Collection & Preprocessing ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_text = build_cleaner()

    # Load CSVs
    jobs = pd.read_csv("job_title_des.csv")
    glass = pd.read_csv("glassdoor_reviews.csv")

    if "Job Description" not in jobs.columns:
        raise KeyError(f"'Job Description' not found. Available: {list(jobs.columns)[:12]}")

    job_text = jobs["Job Description"].dropna().astype(str)

    if all(c in glass.columns for c in ["pros","cons","advice_to_mgmt"]):
        glass_text = (
            glass["pros"].fillna("").astype(str) + " " +
            glass["cons"].fillna("").astype(str) + " " +
            glass["advice_to_mgmt"].fillna("").astype(str)
        ).str.strip()
    else:
        fallback = [c for c in glass.columns if "text" in c.lower() or "review" in c.lower()]
        if not fallback:
            raise KeyError("No suitable text column in glassdoor_reviews.csv")
        glass_text = glass[fallback[0]].dropna().astype(str)

    corpus = pd.concat([job_text, glass_text], ignore_index=True)
    if DOC_LIMIT:
        corpus = corpus.head(DOC_LIMIT)
        print(f"[Phase 1] DOC_LIMIT={DOC_LIMIT} (total {len(job_text)+len(glass_text):,})")

    sentences = [clean_text(doc) for doc in corpus if isinstance(doc, str)]
    print(f"[Phase 1] Processed {len(sentences):,} documents.")

    (OUT_DIR / "clean_sentences.jsonl").write_text(
        "\n".join(json.dumps(s) for s in sentences), encoding="utf-8"
    )
    (OUT_DIR / "bias_word_sets.json").write_text(json.dumps(BIAS_SETS, indent=2), encoding="utf-8")
    return sentences


# Phase 2 — Word2Vec

def phase2_embeddings(sentences):
    print("\n Phase 2: Word Embedding Model (Stronger) ")
    from gensim.models import Word2Vec  # imported here so the bias maths can be tested without gensim
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = Word2Vec(sentences=sentences, **W2V_PARAMS)
    model.save(str(OUT_DIR / "word2vec.model"))
    model.wv.save(str(OUT_DIR / "word2vec.kv"))
    print("[Phase 2] Word2Vec model trained and saved.")
    return model


# Helpers: WEAT & Directions

def mean_vec_unit(kv, words):
    vecs = []
    for w in words:
        if w in kv:
            v = kv[w]; n = np.linalg.norm(v)
            if n > 0: vecs.append(v / n)
    return np.mean(np.stack(vecs, 0), 0) if vecs else None

def cosine(a, b):
    if a is None or b is None: return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return np.nan
    return float(np.dot(a, b) / (na * nb))

def association(kv, w, A, B):
    if w not in kv: return np.nan
    return cosine(kv[w], mean_vec_unit(kv, A)) - cosine(kv[w], mean_vec_unit(kv, B))

def weat_effect_size(kv, X, Y, A, B):
    sX = [association(kv, x, A, B) for x in X if x in kv]
    sY = [association(kv, y, A, B) for y in Y if y in kv]
    if not sX or not sY: return float("nan")
    mX, mY = np.nanmean(sX), np.nanmean(sY)
    sd = np.nanstd(sX + sY, ddof=1)
    return float((mX - mY) / (sd if sd > 0 else np.nan))

def weat_p_permutation(kv, X, Y, A, B, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X = [x for x in X if x in kv]; Y = [y for y in Y if y in kv]
    if len(X) == 0 or len(Y) == 0: return float("nan")
    def mean_assoc(words): return np.nanmean([association(kv, w, A, B) for w in words])
    observed = abs(mean_assoc(X) - mean_assoc(Y))
    pool = np.array(X + Y, dtype=object)
    count = 0
    for _ in range(n):
        rng.shuffle(pool)
        Xp, Yp = pool[:len(X)], pool[len(X):]
        val = abs(mean_assoc(Xp) - mean_assoc(Yp))
        if val >= observed: count += 1
    return float((count + 1) / (n + 1))

def build_unsupervised_direction(kv, male, female):
    parts = []
    m_avg = mean_vec_unit(kv, male)
    f_avg = mean_vec_unit(kv, female)
    if m_avg is not None and f_avg is not None:
        d = m_avg - f_avg
        d /= (np.linalg.norm(d) + 1e-12)
        parts.append(d)
    for a, b in ANCHOR_PAIRS:
        if a in kv and b in kv:
            d = kv[a] - kv[b]
            d /= (np.linalg.norm(d) + 1e-12)
            parts.append(d)
    if not parts:
        return None
    g = np.mean(np.stack(parts, 0), 0)
    g /= (np.linalg.norm(g) + 1e-12)
    return g

def build_supervised_direction(kv, male, female, C=1.0):
    X, y = [], []
    for w in male:
        if w in kv: X.append(kv[w]); y.append(1)
    for w in female:
        if w in kv: X.append(kv[w]); y.append(0)
    if len(X) < 6:  # not enough to train
        return None, np.nan
    X = np.array(X)
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    w = clf.coef_.ravel()
    w /= (np.linalg.norm(w) + 1e-12)
    # training accuracy (on this small lexicon)
    acc = float(clf.score(X, y))
    return w, acc

def combined_direction(kv, male, female, C=3.0):
    """Average of the anchored and the supervised direction, oriented male-positive."""
    parts = [g for g in (build_unsupervised_direction(kv, male, female),
                         build_supervised_direction(kv, male, female, C=C)[0]) if g is not None]
    if not parts:
        return None
    return orient_direction(kv, np.mean(np.stack(parts, 0), 0))


def held_out_lexicon_accuracy(kv, male, female):
    """Leave-one-pair-out check of the gender direction.

    For each (male, female) word pair the direction is rebuilt WITHOUT that pair
    (and without anchor pairs that contain either word), then the two held-out
    words are scored. Scoring words that were used to build the direction, as
    "validation_accuracy" does, is circular and close to 100% by construction.
    """
    global ANCHOR_PAIRS
    pairs = [(m, f) for m, f in zip(male, female) if m in kv and f in kv]
    correct = total = 0
    saved = ANCHOR_PAIRS
    try:
        for m, f in pairs:
            ANCHOR_PAIRS = [(a, b) for a, b in saved if not {a, b} & {m, f}]
            g = combined_direction(kv, [w for w in male if w != m], [w for w in female if w != f])
            if g is None:
                continue
            correct += int(float(np.dot(kv[m], g)) > 0) + int(float(np.dot(kv[f], g)) < 0)
            total += 2
    finally:
        ANCHOR_PAIRS = saved
    return (correct / total if total else float("nan")), total


def orient_direction(kv, g):
    """Flip so that he>0 and she<0 when available."""
    if g is None: return None
    if "he" in kv and "she" in kv:
        if float(np.dot(kv["he"], g)) <= float(np.dot(kv["she"], g)):
            g = -g
    return g / (np.linalg.norm(g) + 1e-12)


# Phase 3 — Bias Measurement

def phase3_bias(model):
    print("\n Phase 3: Bias Measurement ")
    kv = model.wv
    male, female = BIAS_SETS["male"], BIAS_SETS["female"]
    career, family = BIAS_SETS["career"], BIAS_SETS["family"]

    # WEAT
    d = weat_effect_size(kv, male, female, career, family)
    p = weat_p_permutation(kv, male, female, career, family, n=1000)
    print(f"[WEAT] Cohen's d = {d:.3f}, permutation p = {p:.4f}")

    #  Report target/attribute words that are missing from the vocabulary
    missing = {name: [w for w in words if w not in kv] for name, words in BIAS_SETS.items()}
    missing["jobs"] = [w for w in JOB_WORDS if w not in kv]
    for name, words in missing.items():
        if words:
            print(f"[Vocabulary] {name}: not in the corpus, skipped: {', '.join(words)}")

    #  Build multiple directions 
    _, sup_train_acc = build_supervised_direction(kv, male, female, C=3.0)
    g = combined_direction(kv, male, female, C=3.0)
    if g is None:
        raise ValueError("Could not build any gender direction.")

    #  Validation on lexicon (unseen from training? here small set is OK) 
    def project(w): return float(np.dot(kv[w], g)) if w in kv else np.nan
    correct = total = 0
    per_word = []
    for w in male:
        if w in kv:
            s = project(w); total += 1; ok = (s > 0); correct += ok
            per_word.append({"word": w, "score": s, "expected": "male(+)", "correct": bool(ok)})
    for w in female:
        if w in kv:
            s = project(w); total += 1; ok = (s < 0); correct += ok
            per_word.append({"word": w, "score": s, "expected": "female(-)", "correct": bool(ok)})
    acc = correct / total if total else float("nan")
    print(f"[Validation] In-sample lexicon accuracy: {acc:.2%} ({correct}/{total})  (circular: these words built the direction)")
    held_out_acc, held_out_n = held_out_lexicon_accuracy(kv, male, female)
    print(f"[Validation] Held-out lexicon accuracy (leave-one-pair-out): {held_out_acc:.2%} over {held_out_n} words")
          
    #  Project job words 
    rows = []
    for w in JOB_WORDS:
        if w in kv:
            # "cosine" is length-normalised, so it is not inflated for frequent words
            rows.append({"word": w, "projection": project(w), "cosine": cosine(kv[w], g)})
    proj_df = pd.DataFrame(rows, columns=["word", "projection", "cosine"]).sort_values("projection", ascending=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proj_df.to_csv(OUT_DIR / "gender_projection_jobs.csv", index=False)

    # Save metrics
    metrics = {
        "weat_effect_size": float(d),
        "weat_p_value": float(p),
        "validation_accuracy": float(acc),
        "held_out_accuracy": float(held_out_acc),
        "held_out_words": int(held_out_n),
        "missing_words": missing,
        "supervised_train_accuracy": float(sup_train_acc) if not np.isnan(sup_train_acc) else None
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


    return proj_df

# Phase 4 — Visualization

def phase4_visuals(proj_df):
    print("\n Phase 4: Visualization & Reporting ")
    if proj_df.empty:
        print("[Phase 4] No job words available to plot.")
        return

    plt.figure(figsize=(10, 5))
    words = proj_df["word"].tolist()
    vals  = proj_df["projection"].tolist()
    cols  = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    plt.bar(words, vals, color=cols)
    plt.axhline(0, lw=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Projection score (male − female axis)")
    plt.title("Job Words Projected on Gender Direction ")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bar_gender_projection_jobs.png", dpi=180)
    plt.close()
    print(f"Wrote {(OUT_DIR / 'bar_gender_projection_jobs.png').resolve()}")

    # Top +/- charts
    topk = min(10, len(proj_df))
    plt.figure(figsize=(10, 5))
    sub = proj_df.head(topk)
    plt.bar(sub["word"], sub["projection"], color=["#2ca02c"]*len(sub))
    plt.axhline(0, lw=1); plt.xticks(rotation=45, ha="right")
    plt.ylabel("Projection"); plt.title(f"Top +{topk} male-coded job words")
    plt.tight_layout(); plt.savefig(OUT_DIR / "bar_top_male_coded.png", dpi=180); plt.close()

    plt.figure(figsize=(10, 5))
    sub = proj_df.tail(topk).iloc[::-1]
    plt.bar(sub["word"], sub["projection"], color=["#d62728"]*len(sub))
    plt.axhline(0, lw=1); plt.xticks(rotation=45, ha="right")
    plt.ylabel("Projection"); plt.title(f"Top −{topk} female-coded job words")
    plt.tight_layout(); plt.savefig(OUT_DIR / "bar_top_female_coded.png", dpi=180); plt.close()

# 
# Main
if __name__ == "__main__":
    sentences = phase1_preprocessing()
    model = phase2_embeddings(sentences)
    proj_df = phase3_bias(model)
    phase4_visuals(proj_df)
    print("\n Done. Outputs saved in 'outputs/' ")
