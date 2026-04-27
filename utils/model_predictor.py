"""
model_predictor.py — Disease prediction using the trained ensemble.
All source/method strings are internal only; nothing ML-related leaks to responses.
"""
from __future__ import annotations

import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


class DiseasePredictor:
    def __init__(self):
        self.model                    = None
        self.label_encoder            = None
        self.symptom_columns          = None
        self._normalized_feature_names: list[str] = []
        self._feature_lookup:           dict[str, int] = {}
        self.centroid_diseases:         list[str] = []
        self.centroid_disease_to_idx:   dict[str, int] = {}
        self.centroid_matrix            = None
        self.centroid_norms             = None
        self.class_priors               = None
        self.precautions_map:           dict = {}
        self._load_model()

    # ── Load artifacts ────────────────────────────────────────────────────────
    def _load_model(self):
        model_path    = MODEL_DIR / "disease_model.pkl"
        le_path       = MODEL_DIR / "label_encoder.pkl"
        feat_path     = MODEL_DIR / "symptom_features.pkl"
        centroid_path = MODEL_DIR / "disease_centroids.pkl"
        prec_path     = MODEL_DIR / "disease_precautions.pkl"

        if not model_path.exists():
            print("Disease model not found. Run train_model.py first.")
            return

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(le_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        with open(feat_path, "rb") as f:
            self.symptom_columns = pickle.load(f)

        self._normalized_feature_names = [self._norm(c) for c in self.symptom_columns]
        self._feature_lookup = {
            name: idx
            for idx, name in enumerate(self._normalized_feature_names)
            if name
        }

        if prec_path.exists():
            with open(prec_path, "rb") as f:
                self.precautions_map = pickle.load(f)

        if centroid_path.exists():
            with open(centroid_path, "rb") as f:
                cp = pickle.load(f)
            self.centroid_diseases        = cp.get("diseases", [])
            self.centroid_disease_to_idx  = {str(n): i for i, n in enumerate(self.centroid_diseases)}
            raw_centroids                 = cp.get("centroids", None)
            class_counts                  = cp.get("class_counts", None)
            if raw_centroids is not None:
                self.centroid_matrix = np.asarray(raw_centroids, dtype=np.float32)
                self.centroid_norms  = np.linalg.norm(self.centroid_matrix, axis=1) + 1e-12
            if class_counts is not None:
                cc = np.asarray(class_counts, dtype=np.float32)
                if cc.sum() > 0:
                    self.class_priors = cc / cc.sum()

        n = getattr(self.model, "n_features_in_", len(self.symptom_columns) if self.symptom_columns else 0)
        print(f"Disease model loaded. Expects {n} features.")

    # ── Normalisation ─────────────────────────────────────────────────────────
    def _norm(self, value: str) -> str:
        text = (value or "").lower().replace("_", " ").replace("-", " ").replace(".", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ── Feature vector builder ────────────────────────────────────────────────
    def _build_feature_vector(
        self, symptoms_list: list[str]
    ) -> tuple[np.ndarray, int, list[str]]:
        vector        = np.zeros(len(self.symptom_columns), dtype=np.int8)
        matched_count = 0
        matched_names = []
        seen_idx: set[int] = set()

        normalized = [self._norm(s) for s in symptoms_list if s]
        normalized = [s for s in normalized if s]

        for symptom in normalized:
            # 1. Exact lookup
            direct_idx = self._feature_lookup.get(symptom)
            if direct_idx is not None and direct_idx not in seen_idx:
                vector[direct_idx] = 1
                seen_idx.add(direct_idx)
                matched_count += 1
                matched_names.append(self.symptom_columns[direct_idx])
                continue

            # 2. Substring / token overlap / sequence match
            symptom_tokens = set(symptom.split())
            best_idx   = None
            best_score = 0.0
            for idx, feature_name in enumerate(self._normalized_feature_names):
                if not feature_name:
                    continue
                if feature_name == symptom:
                    score = 1.0
                elif feature_name in symptom or symptom in feature_name:
                    score = 0.95
                else:
                    feature_tokens = set(feature_name.split())
                    overlap        = len(symptom_tokens & feature_tokens)
                    if overlap == 0:
                        continue
                    token_score = overlap / max(1, len(symptom_tokens | feature_tokens))
                    ratio_score = SequenceMatcher(None, symptom, feature_name).ratio()
                    score       = max(token_score, ratio_score)
                if score > best_score:
                    best_score = score
                    best_idx   = idx

            # Lowered threshold from 0.60 to 0.50 for better partial matches
            if best_idx is not None and best_score >= 0.50 and best_idx not in seen_idx:
                vector[best_idx] = 1
                seen_idx.add(best_idx)
                matched_count += 1
                matched_names.append(self.symptom_columns[best_idx])

        return vector.reshape(1, -1), matched_count, matched_names

    # ── Main prediction entry point ───────────────────────────────────────────
    def predict(self, symptoms_list: list, user_text: str = "") -> dict:
        if not symptoms_list or self.model is None:
            return self._fallback(symptoms_list, user_text)

        vector, matched_feature_count, matched_names = self._build_feature_vector(symptoms_list)

        # Pad / trim to match model expectations
        expected = self.model.n_features_in_
        if vector.shape[1] != expected:
            if vector.shape[1] < expected:
                vector = np.pad(vector, ((0, 0), (0, expected - vector.shape[1])))
            else:
                vector = vector[:, :expected]

        # ── Primary: ensemble predict_proba ──────────────────────────────────
        probs      = self.model.predict_proba(vector)[0]
        sorted_idx = np.argsort(probs)[::-1]

        rf_confidence = float(probs[sorted_idx[0]])
        rf_second     = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        rf_margin     = rf_confidence - rf_second
        pred_idx      = self.model.predict(vector)[0]
        rf_disease    = self.label_encoder.inverse_transform([pred_idx])[0]

        disease            = rf_disease
        confidence         = rf_confidence
        confidence_margin  = rf_margin
        method             = "ensemble"

        # Build top-3 alternatives from ensemble proba
        alternatives = []
        for i in sorted_idx[1:4]:
            if float(probs[i]) > 0.01:
                alt_name = self.label_encoder.inverse_transform([i])[0]
                alternatives.append({"disease": alt_name, "confidence": float(probs[i])})

        # ── Secondary: Bayes / centroid refinement ────────────────────────────
        # KEY FIX: Only allow Bayes to override ensemble when it has MUCH stronger signal.
        # Old logic was overriding ensemble on ≤2 features (too aggressive).
        # New logic: Bayes only overrides if it has confidence > ensemble + 0.15 margin.
        bayes_disease     = disease
        bayes_conf        = 0.0
        bayes_margin      = 0.0
        bayes_alternatives: list[dict] = []

        if self.centroid_matrix is not None and len(self.centroid_diseases) > 0:
            x       = vector.astype(np.float32)[0]
            pos_idx = np.where(x > 0)[0]

            if len(pos_idx) > 0:
                freq    = np.clip(self.centroid_matrix[:, pos_idx], 1e-4, 1 - 1e-4)
                log_ll  = np.sum(np.log(freq), axis=1)
                if self.class_priors is not None and len(self.class_priors) == len(log_ll):
                    log_ll = log_ll + 0.4 * np.log(np.clip(self.class_priors, 1e-8, 1.0))
                ll_max = float(np.max(log_ll))
                exp_s  = np.exp(log_ll - ll_max)
                norm_s = exp_s / (np.sum(exp_s) + 1e-12)

                b_idx  = np.argsort(norm_s)[::-1]
                top_b  = int(b_idx[0])

                bayes_disease = self.centroid_diseases[top_b]
                bayes_conf    = float(norm_s[top_b])
                bayes_second  = float(norm_s[b_idx[1]]) if len(b_idx) > 1 else 0.0
                bayes_margin  = bayes_conf - bayes_second
                for i in b_idx[1:4]:
                    if norm_s[i] > 0.005:
                        bayes_alternatives.append(
                            {"disease": self.centroid_diseases[int(i)], "confidence": float(norm_s[i])}
                        )

                # KEY FIX: Only override ensemble if Bayes is substantially more confident.
                # Removed the old "matched_feature_count <= 2" override — this was causing
                # the model to pick wrong diseases when few features matched.
                # Now Bayes only overrides if it's clearly better (>15% margin advantage).
                if bayes_conf > confidence + 0.15:
                    disease            = bayes_disease
                    confidence         = bayes_conf
                    confidence_margin  = bayes_margin
                    method             = "bayes_centroid"
                    alternatives       = bayes_alternatives

            # Cosine fallback (only for 3+ matched features)
            x_norm = float(np.linalg.norm(x))
            if x_norm > 0 and matched_feature_count >= 3:
                sims  = (self.centroid_matrix @ x) / (self.centroid_norms * x_norm)
                c_idx = np.argsort(sims)[::-1]
                top_c = int(c_idx[0])
                c_conf = float(sims[top_c])
                # Only override if cosine is clearly better than current
                if c_conf > confidence + 0.08:
                    disease            = self.centroid_diseases[top_c]
                    confidence         = c_conf
                    confidence_margin  = c_conf - float(sims[c_idx[1]]) if len(c_idx) > 1 else c_conf
                    method             = "centroid_cosine"
                    alternatives       = [
                        {"disease": self.centroid_diseases[int(i)], "confidence": float(sims[i])}
                        for i in c_idx[1:4] if sims[i] > 0
                    ]

        # ── Coverage score ───────────────────────────────────────────────────
        coverage_score = 0.0
        if self.centroid_matrix is not None:
            d_idx = self.centroid_disease_to_idx.get(str(disease))
            if d_idx is not None:
                active = np.where(vector[0] > 0)[0]
                if len(active) > 0:
                    coverage_score = float(np.mean(self.centroid_matrix[d_idx, active]))

        # ── Urgency ──────────────────────────────────────────────────────────
        urgency    = "low"
        urgent_syms = {"chest pain", "shortness of breath", "difficulty breathing", "bleeding"}
        medium_syms = {"fever", "vomiting", "diarrhea", "rash", "headache", "body aches", "nausea"}
        if any(s in urgent_syms for s in symptoms_list):
            urgency = "high"
        elif any(s in medium_syms for s in symptoms_list):
            urgency = "medium"

        # ── Reliability gates ────────────────────────────────────────────────
        # KEY FIX: Lower the confidence thresholds slightly to allow more responses
        # to show through, especially for 2-symptom inputs.
        if matched_feature_count >= 4:
            ml_reliable       = confidence >= 0.10 and confidence_margin >= 0.003
            response_eligible = confidence >= 0.04
        elif matched_feature_count == 3:
            ml_reliable       = confidence >= 0.10 and confidence_margin >= 0.004
            response_eligible = confidence >= 0.04
        elif matched_feature_count == 2:
            ml_reliable       = confidence >= 0.07 and confidence_margin >= 0.003
            response_eligible = confidence >= 0.02
        elif matched_feature_count == 1:
            ml_reliable       = confidence >= 0.12 and confidence_margin >= 0.01
            response_eligible = confidence >= 0.06
        else:
            ml_reliable       = False
            response_eligible = False

        precautions = self.precautions_map.get(disease, ["Consult a doctor for proper diagnosis."])

        return {
            "disease":               disease,
            "confidence":            confidence,
            "confidence_margin":     confidence_margin,
            "method":                method,          # internal only
            "ml_reliable":           ml_reliable,
            "response_eligible":     response_eligible,
            "matched_feature_count": matched_feature_count,
            "matched_feature_names": matched_names,
            "coverage_score":        coverage_score,
            "precautions":           precautions,
            "summary":               "",              # blank — no ML wording
            "matched_symptoms":      symptoms_list,
            "alternatives":          alternatives,
            "urgency":               urgency,
            "source":                "symptom checker",
            "user_text":             user_text or "",
        }

    # ── Fallback ──────────────────────────────────────────────────────────────
    def _fallback(self, symptoms, user_text):
        return {
            "disease":               "Could not identify a specific condition",
            "confidence":            0.0,
            "confidence_margin":     0.0,
            "method":                "none",
            "ml_reliable":           False,
            "response_eligible":     False,
            "matched_feature_count": 0,
            "matched_feature_names": [],
            "coverage_score":        0.0,
            "precautions":           ["Describe 2-3 clear symptoms.", "If severe, consult a doctor."],
            "summary":               "",
            "matched_symptoms":      symptoms or [],
            "alternatives":          [],
            "urgency":               "low",
            "source":                "symptom checker",
            "user_text":             user_text or "",
        }