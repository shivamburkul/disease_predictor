"""
qa_matcher.py — Hybrid word + char TF-IDF retrieval for MedQuAD.
Char n-grams handle typos/abbreviations; word n-grams handle semantics.
"""
from __future__ import annotations

import csv
import os
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from utils.medical_knowledge import normalize_text, tokenize

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


class QAMatcher:
    # Minimum cosine similarity to return a result
    THRESHOLD = 0.12

    def __init__(self, csv_path: str = None):
        if csv_path is None:
            csv_path = str(BASE_DIR / "data" / "medquad.csv")
        self.csv_path = csv_path

        self.entries:      list[dict]              = []
        self.questions:    list[str]               = []
        self.enriched:     list[str]               = []
        self.answers:      list[str]               = []
        self.focus_areas:  list[str]               = []

        self.word_vec: TfidfVectorizer | None      = None
        self.char_vec: TfidfVectorizer | None      = None
        self.question_matrix                       = None   # sparse hybrid matrix

        self._canonical_to_idx: dict[str, int]    = {}
        self._load_or_build_model()

    # ── Canonical normalization ───────────────────────────────────────────────
    def _canonical(self, text: str) -> str:
        base = normalize_text(text)
        base = re.sub(r"\bwhat\s*\(\s*are\s*\)\b", "what", base)
        base = re.sub(r"\bwhat\s*\(\s*is\s*\)\b",  "what", base)
        base = re.sub(r"\s+", " ", base).strip().rstrip("?.! ")
        return base

    # ── Load CSV ──────────────────────────────────────────────────────────────
    def _load_csv_entries(self) -> list[dict]:
        if not os.path.exists(self.csv_path):
            print(f"Warning: {self.csv_path} not found. Q&A model disabled.")
            return []
        entries = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
                if first.startswith("version https://git-lfs"):
                    print("Warning: medquad.csv is an LFS pointer.")
                    return []
                f.seek(0)
                reader = csv.DictReader(f)
                for row in reader:
                    q     = (row.get("question")   or row.get("Question")   or "").strip()
                    a     = (row.get("answer")     or row.get("Answer")     or "").strip()
                    focus = (row.get("focus_area") or row.get("Focus_area") or "").strip()
                    if q and a:
                        enriched_q = f"{q} {focus} {focus}".strip() if focus else q
                        entries.append({
                            "question":          q,
                            "enriched_question": enriched_q,
                            "answer":            a,
                            "focus_area":        focus,
                        })
        except OSError as exc:
            print(f"Error loading medquad.csv: {exc}")
        return entries

    # ── Load / build model ────────────────────────────────────────────────────
    def _load_or_build_model(self) -> None:
        word_path   = MODEL_DIR / "qa_word_vectorizer.pkl"
        char_path   = MODEL_DIR / "qa_char_vectorizer.pkl"
        entries_path = MODEL_DIR / "qa_entries.pkl"

        csv_entries = self._load_csv_entries()
        csv_count   = len(csv_entries)

        # Use saved artifacts if they match current CSV
        if word_path.exists() and entries_path.exists():
            try:
                with open(entries_path, "rb") as f:
                    stored = pickle.load(f)
                if isinstance(stored, list) and len(stored) == csv_count and csv_count > 0:
                    with open(word_path, "rb") as f:
                        self.word_vec = pickle.load(f)
                    if char_path.exists():
                        with open(char_path, "rb") as f:
                            self.char_vec = pickle.load(f)
                    self.entries = stored
                    self._build_index()
                    return
            except Exception:
                pass

        # Fall back to legacy single vectorizer if new ones not found
        legacy_path = MODEL_DIR / "qa_vectorizer.pkl"
        if legacy_path.exists() and entries_path.exists():
            try:
                with open(entries_path, "rb") as f:
                    stored = pickle.load(f)
                if isinstance(stored, list) and len(stored) == csv_count and csv_count > 0:
                    with open(legacy_path, "rb") as f:
                        self.word_vec = pickle.load(f)
                    self.entries = stored
                    self._build_index()
                    return
            except Exception:
                pass

        # Build fresh from CSV
        self.entries = csv_entries
        if not self.entries:
            return
        self._fit_vectorizers()
        self._build_index()

    def _fit_vectorizers(self) -> None:
        enriched = [e["enriched_question"] for e in self.entries]

        self.word_vec = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 3),
            stop_words="english",
            min_df=1,
            max_df=0.90,
            sublinear_tf=True,
            analyzer="word",
        )
        self.word_vec.fit(enriched)

        self.char_vec = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            analyzer="char_wb",
            max_features=80_000,
        )
        self.char_vec.fit(enriched)

    def _build_index(self) -> None:
        self.questions   = [e["question"]   for e in self.entries]
        self.enriched    = [e.get("enriched_question", e["question"]) for e in self.entries]
        self.answers     = [e["answer"]     for e in self.entries]
        self.focus_areas = [e.get("focus_area", "") for e in self.entries]

        if self.word_vec is not None:
            W = self.word_vec.transform(self.enriched)
            if self.char_vec is not None:
                C = self.char_vec.transform(self.enriched)
                self.question_matrix = hstack([W, C])
            else:
                self.question_matrix = W

        # Canonical exact-match lookup
        self._canonical_to_idx = {}
        for i, q in enumerate(self.questions):
            key = self._canonical(q)
            if key not in self._canonical_to_idx:
                self._canonical_to_idx[key] = i

    # ── Query encoding ────────────────────────────────────────────────────────
    def _encode_query(self, text: str):
        """Return hybrid sparse vector matching question_matrix columns."""
        W = self.word_vec.transform([text])
        if self.char_vec is not None:
            C = self.char_vec.transform([text])
            return hstack([W, C])
        return W

    # ── Main retrieval ────────────────────────────────────────────────────────
    def get_answer(self, user_query: str, threshold: float = None) -> dict | None:
        if not self.entries or self.word_vec is None or self.question_matrix is None:
            return None

        query = normalize_text(user_query)
        if not query:
            return None

        effective_threshold = threshold if threshold is not None else self.THRESHOLD

        # 1. Exact canonical match
        canonical_query = self._canonical(user_query)
        exact_idx = self._canonical_to_idx.get(canonical_query)
        if exact_idx is not None:
            return self._make_result(exact_idx, score=1.0, margin=1.0, query=query)

        # 2. Hybrid TF-IDF cosine similarity
        q_vec = self._encode_query(query)
        sims  = (q_vec @ self.question_matrix.T).toarray()[0]

        sorted_sims  = np.sort(sims)[::-1]
        best_score   = float(sorted_sims[0])
        second_score = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
        margin       = best_score - second_score
        best_idx     = int(np.argmax(sims))

        if best_score >= effective_threshold and margin >= 0.003:
            return self._make_result(best_idx, score=best_score, margin=margin, query=query)

        # 3. Fuzzy string fallback
        best_fuzzy_idx = -1
        best_fuzzy     = 0.0
        for i, q in enumerate(self.questions):
            ratio = SequenceMatcher(None, canonical_query, self._canonical(q)).ratio()
            if ratio > best_fuzzy:
                best_fuzzy     = ratio
                best_fuzzy_idx = i

        if best_fuzzy_idx >= 0 and best_fuzzy >= 0.82:
            return self._make_result(
                best_fuzzy_idx,
                score=best_fuzzy,
                margin=max(0.0, best_fuzzy - 0.5),
                query=query,
            )

        return None

    def _make_result(self, idx: int, score: float, margin: float, query: str) -> dict:
        q_terms = set(tokenize(query))
        matched = [t for t in tokenize(normalize_text(self.questions[idx])) if t in q_terms]
        matched = list(dict.fromkeys(matched))[:8]
        return {
            "question":      self.questions[idx],
            "answer":        self.answers[idx],
            "score":         min(1.0, score),
            "matched_terms": matched,
            "source":        "medquad.csv",
            "margin":        margin,
            "focus_area":    self.focus_areas[idx] if idx < len(self.focus_areas) else "",
        }