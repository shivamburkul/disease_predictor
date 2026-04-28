from __future__ import annotations

import csv
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

SYMPTOM_DATA_PATH = DATA_DIR / "symptoms_diseases.csv"
MEDQUAD_PATH      = DATA_DIR / "medquad.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_real_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip().lower()
    if first.startswith("version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path.name} is a Git LFS pointer. Run 'git lfs pull' first."
        )


def _detect_disease_column(columns: list[str]) -> str:
    for col in columns:
        low = col.lower().strip()
        if low in {"disease", "diseases", "condition", "conditions"} or "disease" in low:
            return col
    raise ValueError(f"No disease/condition column found in: {columns[:15]}")


def _stratified_split_with_singletons(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train/test split that handles singleton classes safely.

    Key fix: re-encode y to compact 0..K-1 with LabelEncoder before
    np.bincount. This handles the case where y is a subset of a larger
    label space (some integers between 0 and max(y) may be absent),
    which would give incorrect singleton counts with raw np.bincount.
    """
    # Compact re-encoding so np.bincount produces correct dense counts
    compact_enc = LabelEncoder()
    y_compact   = compact_enc.fit_transform(y)

    counts           = np.bincount(y_compact)
    singleton_mask   = counts == 1
    is_singleton_row = singleton_mask[y_compact]
    is_normal_row    = ~is_singleton_row

    # Singleton rows forced to train (model must see every class)
    X_sing = X[is_singleton_row]
    y_sing = y[is_singleton_row]

    # Normal rows safe to stratify
    X_norm         = X[is_normal_row]
    y_norm         = y[is_normal_row]
    y_norm_compact = y_compact[is_normal_row]   # compact labels for stratify=

    X_tr_norm, X_test, y_tr_norm, y_test = train_test_split(
        X_norm, y_norm,
        test_size=test_size,
        random_state=random_state,
        stratify=y_norm_compact,   # always safe: every class has >= 2 samples here
    )

    X_train = np.concatenate([X_tr_norm, X_sing], axis=0)
    y_train = np.concatenate([y_tr_norm, y_sing], axis=0)

    n_sing = int(singleton_mask.sum())
    n_norm = int((~singleton_mask).sum())
    print(f"  Singleton classes forced to train-only : {n_sing}")
    print(f"  Classes with stratified split          : {n_norm}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Disease model
# ─────────────────────────────────────────────────────────────────────────────

def train_disease_model(evaluate: bool = True) -> None:
    print("=" * 60)
    print("TRAINING DISEASE MODEL")
    print("=" * 60)
    _ensure_real_csv(SYMPTOM_DATA_PATH)

    # Read header to discover columns
    header_df    = pd.read_csv(SYMPTOM_DATA_PATH, nrows=0)
    disease_col  = _detect_disease_column(header_df.columns.tolist())
    feature_cols = [c for c in header_df.columns if c != disease_col]
    print(f"Disease column : '{disease_col}'")
    print(f"Feature columns: {len(feature_cols)}")

    # Load dataset in chunks to save RAM
    print("Reading dataset in chunks (int8 dtype to save RAM)...")
    chunks = []
    for chunk in pd.read_csv(
        SYMPTOM_DATA_PATH,
        usecols=[disease_col] + feature_cols,
        chunksize=30_000,
        low_memory=True,
    ):
        chunk[disease_col] = chunk[disease_col].astype(str).str.strip()
        chunk = chunk[
            chunk[disease_col].notna()
            & (chunk[disease_col] != "")
            & (chunk[disease_col] != "nan")
        ]
        chunk[feature_cols] = (
            chunk[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(np.int8)
        )
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    print(f"Loaded: {len(df)} rows")

    X      = df[feature_cols].values.astype(np.int8)
    y_text = df[disease_col].values
    del df
    gc.collect()

    encoder = LabelEncoder()
    y       = encoder.fit_transform(y_text)
    del y_text
    gc.collect()

    n_classes  = len(encoder.classes_)
    n_features = len(feature_cols)
    print(f"Classes : {n_classes}")
    print(f"Features: {n_features}")
    print(f"Rows    : {len(y)}")

    # Disease centroids (used at inference by DiseasePredictor)
    print("\nComputing disease centroids...")
    centroid_sums = np.zeros((n_classes, n_features), dtype=np.float32)
    class_counts  = np.zeros(n_classes, dtype=np.int32)
    np.add.at(centroid_sums, y, X.astype(np.float32))
    np.add.at(class_counts,  y, 1)
    safe_counts = np.maximum(class_counts.astype(np.float32), 1.0)
    centroids   = centroid_sums / safe_counts[:, None]
    del centroid_sums
    gc.collect()

    # Main train / test split
    print("\nSplitting dataset (singleton-safe stratified split)...")
    X_train, X_test, y_train, y_test = _stratified_split_with_singletons(
        X, y, test_size=0.2, random_state=42
    )
    del X, y
    gc.collect()
    print(f"Train : {len(X_train)} rows")
    print(f"Test  : {len(X_test)} rows")

    # [1/2] ExtraTreesClassifier
    # n_jobs=-1 uses all CPU cores (~3x faster than n_jobs=1)
    print("\n[1/2] Training ExtraTreesClassifier (150 trees, all cores)...")
    et = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    et.fit(X_train, y_train)
    print("      ExtraTrees done.")
    gc.collect()

    # [2/2] RandomForestClassifier
    print("\n[2/2] Training RandomForestClassifier (100 trees, all cores)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=7,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    print("      RandomForest done.")
    gc.collect()

    # Assemble soft-voting ensemble (no re-training needed)
    # Both estimators already output valid probabilities via predict_proba.
    # No calibration step is needed — this eliminates the 30+ min hang.
    print("\nAssembling Voting Ensemble (ExtraTrees + RandomForest)...")
    ensemble = VotingClassifier(
        estimators=[("et", et), ("rf", rf)],
        voting="soft",
        weights=[1, 1],
        n_jobs=1,
    )
    ensemble.estimators_       = [et, rf]
    ensemble.named_estimators_ = {"et": et, "rf": rf}
    ensemble.le_               = LabelEncoder().fit(y_train)
    ensemble.classes_          = ensemble.le_.classes_
    print("Ensemble assembled. Training complete.")

    # Evaluation
    top1 = top3 = None
    if evaluate:
        print("\nEvaluating on test set...")
        y_pred        = ensemble.predict(X_test)
        top1          = float(accuracy_score(y_test, y_pred))
        y_prob        = ensemble.predict_proba(X_test)
        model_classes = ensemble.classes_
        top3_hits     = 0
        for i in range(len(y_test)):
            top3_idx          = np.argsort(y_prob[i])[::-1][:3]
            predicted_classes = model_classes[top3_idx]
            if y_test[i] in predicted_classes:
                top3_hits += 1
        top3 = top3_hits / len(y_test)

        print(f"\n{'='*40}")
        print("DISEASE MODEL ACCURACY")
        print(f"{'='*40}")
        print(f"  Top-1 : {top1:.4f}  ({top1*100:.1f}%)")
        print(f"  Top-3 : {top3:.4f}  ({top3*100:.1f}%)")
        print(f"{'='*40}\n")

    del X_train, X_test, y_train, y_test
    gc.collect()

    # Save model artifacts
    print("Saving model artifacts...")
    with (MODEL_DIR / "disease_model.pkl").open("wb") as f:
        pickle.dump(ensemble, f)
    with (MODEL_DIR / "label_encoder.pkl").open("wb") as f:
        pickle.dump(encoder, f)
    with (MODEL_DIR / "symptom_features.pkl").open("wb") as f:
        pickle.dump(feature_cols, f)
    with (MODEL_DIR / "disease_centroids.pkl").open("wb") as f:
        pickle.dump(
            {
                "diseases":     list(encoder.classes_),
                "centroids":    centroids,
                "class_counts": class_counts.astype(np.int32),
            },
            f,
        )
    with (MODEL_DIR / "disease_precautions.pkl").open("wb") as f:
        pickle.dump({}, f)

    print("Disease model saved.")
    print(f"  Classes : {n_classes}")
    print(f"  Features: {n_features}")
    if top1 is not None:
        print(f"  Top-1   : {top1:.4f}")
        print(f"  Top-3   : {top3:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Q&A model — hybrid word + char TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def train_qa_model(evaluate: bool = True) -> None:
    print("\n" + "=" * 60)
    print("TRAINING Q&A MODEL")
    print("=" * 60)
    _ensure_real_csv(MEDQUAD_PATH)

    entries: list[dict] = []
    with MEDQUAD_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q     = (row.get("question")   or row.get("Question")   or "").strip()
            a     = (row.get("answer")     or row.get("Answer")     or "").strip()
            focus = (row.get("focus_area") or row.get("Focus_area") or "").strip()
            if q and a:
                enriched_q = f"{q} {focus} {focus}".strip() if focus else q
                entries.append(
                    {
                        "question":          q,
                        "enriched_question": enriched_q,
                        "answer":            a,
                        "focus_area":        focus,
                    }
                )

    if not entries:
        raise ValueError("No valid Q&A rows found in medquad.csv")

    print(f"Loaded: {len(entries)} Q&A pairs")

    enriched_questions = [e["enriched_question"] for e in entries]

    word_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 3),
        stop_words="english",
        min_df=1,
        max_df=0.90,
        sublinear_tf=True,
        analyzer="word",
    )
    word_vec.fit(enriched_questions)

    char_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        analyzer="char_wb",
        max_features=80_000,
    )
    char_vec.fit(enriched_questions)

    if evaluate:
        from scipy.sparse import hstack

        print("\nEvaluating Q&A retrieval...")
        sample_size    = min(2000, len(entries))
        rng            = np.random.RandomState(42)
        indices        = rng.choice(len(entries), sample_size, replace=False)
        sample_entries = [entries[i] for i in indices]

        W_all = word_vec.transform(enriched_questions)
        C_all = char_vec.transform(enriched_questions)
        X_all = hstack([W_all, C_all])

        focus_hits = 0
        exact_hits = 0
        for e in sample_entries:
            eq    = e["enriched_question"]
            q_vec = hstack([word_vec.transform([eq]), char_vec.transform([eq])])
            sims  = (q_vec @ X_all.T).toarray()[0]
            q_idx = entries.index(e)
            sims[q_idx] = -1
            top3_idx = np.argsort(sims)[::-1][:3]
            top1_idx = int(top3_idx[0])

            if entries[top1_idx]["answer"] == e["answer"]:
                exact_hits += 1
            if any(entries[j]["focus_area"] == e["focus_area"] for j in top3_idx):
                focus_hits += 1

        exact_acc = exact_hits / sample_size
        focus_acc = focus_hits / sample_size
        print(f"\n{'='*40}")
        print("Q&A MODEL ACCURACY")
        print(f"{'='*40}")
        print(f"  Exact answer match (top-1): {exact_acc:.4f} ({exact_acc*100:.1f}%)")
        print(f"  Same topic match  (top-3) : {focus_acc:.4f} ({focus_acc*100:.1f}%)")
        print(f"  NOTE: Exact match is low because MedQuAD has many")
        print(f"  near-identical questions with different short answers.")
        print(f"  Topic match is the real accuracy for your chatbot.")
        print(f"{'='*40}\n")

    with (MODEL_DIR / "qa_word_vectorizer.pkl").open("wb") as f:
        pickle.dump(word_vec, f)
    with (MODEL_DIR / "qa_char_vectorizer.pkl").open("wb") as f:
        pickle.dump(char_vec, f)
    with (MODEL_DIR / "qa_vectorizer.pkl").open("wb") as f:
        pickle.dump(word_vec, f)
    with (MODEL_DIR / "qa_entries.pkl").open("wb") as f:
        pickle.dump(entries, f)

    print(f"Q&A model saved: {len(entries)} entries")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_disease_model(evaluate=True)
    train_qa_model(evaluate=True)
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("Start the app with: python app.py")
    print("=" * 60)