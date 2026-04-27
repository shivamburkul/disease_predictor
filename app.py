from __future__ import annotations

import csv
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from utils.intent_detector import detect_intent
from utils.medical_knowledge import DEFAULT_PROMPTS, normalize_text
from utils.model_predictor import DiseasePredictor
from utils.qa_matcher import QAMatcher
from utils.response_generator import (
    build_answer_response,
    build_disease_response,
    build_error_response,
    build_greeting_response,
    build_need_more_details_response,
    build_non_medical_response,
    build_pattern_response,
    build_urgent_symptom_response,
)
from utils.symptom_extractor import extract_symptoms, load_symptom_list

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SYMPTOM_CSV = DATA_DIR / "symptoms_diseases.csv"
MEDQUAD_CSV = DATA_DIR / "medquad.csv"

app = Flask(__name__)

# Detect pasted dataset rows (25+ comma-separated items)
DATA_DUMP_RE = re.compile(r"(?:\w[\w\s\-()']*,){25,}\w[\w\s\-()']*")

URGENT_SINGLE_SYMPTOMS = {
    "chest pain", "shortness of breath", "difficulty breathing", "difficulty speaking",
}


def _looks_like_dataset_dump(text: str) -> bool:
    if len(text) > 350 and text.count(",") > 20:
        return True
    return bool(DATA_DUMP_RE.search(text.lower()))


def _csv_is_real(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip().lower()
        return not first.startswith("version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _load_symptoms_from_csv(path: Path) -> list[str]:
    """Load symptom column names from the one-hot CSV as the symptom vocabulary."""
    if not _csv_is_real(path):
        return []
    try:
        import pandas as pd
        header = pd.read_csv(path, nrows=0)
        cols = header.columns.tolist()
        # Find disease column and exclude it
        disease_col = None
        for c in cols:
            low = c.lower().strip()
            if low in {"disease", "diseases", "condition"} or "disease" in low:
                disease_col = c
                break
        symptoms = [c for c in cols if c != disease_col]
        return symptoms
    except Exception:
        return []


# Load symptom vocabulary from CSV columns
csv_symptoms = _load_symptoms_from_csv(SYMPTOM_CSV)
load_symptom_list(csv_symptoms)
print(f"Symptom vocabulary loaded: {len(csv_symptoms)} terms")

predictor = DiseasePredictor()
qa_matcher = QAMatcher(str(MEDQUAD_CSV))


@app.route("/")
def index():
    return render_template("index.html", prompts=DEFAULT_PROMPTS)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        user_message = (payload.get("message") or "").strip()
        if not user_message:
            return jsonify(build_error_response("Please type a message before sending.")), 400

        if _looks_like_dataset_dump(user_message):
            return jsonify(
                build_need_more_details_response(
                    "Please send one sentence describing your symptoms, not a full symptom list."
                )
            )

        symptoms = extract_symptoms(user_message)
        intent = detect_intent(user_message, has_symptoms=bool(symptoms))
        response_payload = _route(user_message, intent, symptoms)
        response_payload["intent"] = intent
        response_payload["matched_symptoms"] = symptoms
        response_payload["message"] = user_message
        return jsonify(response_payload)
    except Exception as exc:
        return jsonify(build_error_response(f"Server error: {exc}")), 500


def _route(user_message: str, intent: str, symptoms: list[str]) -> dict:
    # ── Greeting ──────────────────────────────────────────────────────────────
    if intent == "greeting":
        return build_greeting_response()

    # ── Non-medical ───────────────────────────────────────────────────────────
    if intent == "non_medical":
        return build_non_medical_response(user_message)

    # ── Medical question ──────────────────────────────────────────────────────
    if intent == "question":
        # Try QA model first
        faq = qa_matcher.get_answer(user_message)
        if faq:
            return build_answer_response(faq)

        # Try symptom-based QA fallback (e.g. "what are loose motions")
        if symptoms:
            for s in symptoms[:2]:
                fallback = qa_matcher.get_answer(f"what causes {s}")
                if fallback:
                    return build_answer_response(fallback)
            # Try disease prediction as last resort
            pred = predictor.predict(symptoms, user_text=user_message)
            if pred.get("ml_reliable") or pred.get("response_eligible"):
                return build_disease_response(pred)

        return build_need_more_details_response(
            "I couldn't find information on that. Try asking a specific medical question like "
            "'What causes diabetes?' or 'What are symptoms of malaria?'"
        )

    # ── Symptom report ────────────────────────────────────────────────────────
    if intent == "symptom":
        if not symptoms:
            return build_need_more_details_response(
                "I couldn't detect specific symptoms. Please use clear terms like: "
                "fever, cough, headache, chest pain, vomiting, diarrhea."
            )

        # Urgent single symptoms — always respond immediately
        urgent_hits = [s for s in symptoms if s in URGENT_SINGLE_SYMPTOMS]
        if urgent_hits and len(symptoms) == 1:
            return build_urgent_symptom_response(urgent_hits)

        # Run ML prediction
        pred = predictor.predict(symptoms, user_text=user_message)

        if pred.get("ml_reliable"):
            return build_disease_response(pred)

        if pred.get("response_eligible"):
            return build_disease_response(pred)

        # Even if not reliable, if we matched at least 1 feature, show pattern
        if pred.get("matched_feature_count", 0) >= 1 and pred.get("confidence", 0) >= 0.02:
            return build_pattern_response(pred)

        # QA fallback for single symptom
        for s in symptoms[:2]:
            info = qa_matcher.get_answer(f"what causes {s}")
            if info:
                return build_answer_response(info)

        return build_need_more_details_response(
            "I need more details for a reliable prediction. "
            "Please describe 2-3 symptoms, for example: fever, headache, and cough."
        )

    # ── Unknown intent — try both QA and symptom ─────────────────────────────
    if symptoms:
        pred = predictor.predict(symptoms, user_text=user_message)
        if pred.get("ml_reliable") or pred.get("response_eligible"):
            return build_disease_response(pred)

    faq = qa_matcher.get_answer(user_message)
    if faq:
        return build_answer_response(faq)

    return build_need_more_details_response(
        "Please ask a medical question or describe your symptoms clearly."
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)