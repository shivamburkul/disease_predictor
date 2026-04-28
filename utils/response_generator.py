"""
response_generator.py — Builds all chat response payloads.
No ML/model references are exposed to the user.
"""
from __future__ import annotations

import random
import re


GREETING_VARIANTS = [
    "Hi there! I'm MedBot. Describe your symptoms or ask any medical question — I'm here to help.",
    "Hello! Tell me what you're feeling or ask a health question.",
    "Welcome! You can describe symptoms or ask a medical question.",
]

NON_MEDICAL_RESPONSES = {
    "how are you":     "I'm here and ready to help with medical questions.",
    "who are you":     "I'm MedBot, a medical assistant. How can I help you today?",
    "what are you":    "I'm MedBot, built to answer medical questions and symptom-based queries.",
    "what is your name": "I'm MedBot.",
}


def _ensure_list(value):
    if not isinstance(value, list):
        return []
    return [item for item in value if item is not None and not isinstance(item, float)]


def _base(msg_type: str, title: str, summary: str, **kw) -> dict:
    sections         = kw.pop("sections", [])
    if not isinstance(sections, list):
        sections = []
    confidence       = kw.pop("confidence", None)
    matched_symptoms = _ensure_list(kw.pop("matched_symptoms", []))
    precautions      = _ensure_list(kw.pop("precautions", []))
    alternatives     = kw.pop("alternatives", [])
    if not isinstance(alternatives, list):
        alternatives = []
    urgency = kw.pop("urgency", "normal")
    source  = kw.pop("source", "medical assistant")

    return {
        "type":             msg_type,
        "title":            title,
        "summary":          summary,
        "sections":         sections,
        "confidence":       confidence,
        "matched_symptoms": matched_symptoms,
        "precautions":      precautions,
        "alternatives":     alternatives,
        "urgency":          urgency,
        "source":           source,
        **kw,
    }


def build_greeting_response() -> dict:
    return _base(
        "greeting",
        "MedBot — Medical Assistant",
        random.choice(GREETING_VARIANTS),
        source="greeting",
    )


def build_non_medical_response(user_message: str) -> dict:
    from utils.medical_knowledge import normalize_text

    msg = normalize_text(user_message)
    for key, reply in NON_MEDICAL_RESPONSES.items():
        if key in msg:
            return _base("general", "MedBot", reply, sections=[], source="built-in")

    return _base(
        "general",
        "MedBot",
        "Please ask a medical question or describe symptoms. Non-medical chat is limited.",
        sections=[],
        source="built-in",
    )


def build_need_more_details_response(message: str) -> dict:
    return _base(
        "unknown",
        "Need More Medical Details",
        message,
        sections=[{
            "heading": "Try this format",
            "points": [
                "Mention at least 2 symptoms",
                "Add duration (for example: for 2 days)",
                "Use simple terms like fever, cough, vomiting, chest pain",
            ],
        }],
        source="ml_fallback",
    )


def build_urgent_symptom_response(symptoms: list[str]) -> dict:
    return _base(
        "unknown",
        "Urgent Symptom Detected",
        "Your symptom may require urgent medical attention. Please contact emergency care immediately.",
        sections=[{
            "heading": "Detected urgent symptom(s)",
            "points": symptoms[:5],
        }],
        urgency="high",
        matched_symptoms=symptoms,
        source="urgent_triage",
    )


def build_answer_response(match: dict) -> dict:
    answer   = match.get("answer", "")
    question = match.get("question", "")
    score    = match.get("score", 0.0)
    if isinstance(score, (float, int)):
        score = min(1.0, max(0.0, float(score)))
    else:
        score = 0.0
    sections = _structure_answer(answer)
    return _base(
        "answer",
        "Medical Information",
        _first_sentence(answer),
        sections=sections,
        confidence=score,
        matched_question=question,
        source="Knowledge Base",
    )


def build_disease_response(result: dict) -> dict:
    disease    = result.get("disease", "Possible condition")
    confidence = result.get("confidence", 0.0)
    if isinstance(confidence, (float, int)):
        confidence = min(1.0, max(0.0, float(confidence)))
    else:
        confidence = 0.0

    urgency     = result.get("urgency", "low")
    precautions = result.get("precautions", [])
    if not isinstance(precautions, list):
        precautions = []
    alternatives = result.get("alternatives", [])
    if not isinstance(alternatives, list):
        alternatives = []
    matched = result.get("matched_symptoms", [])
    if not isinstance(matched, list):
        matched = []

    sections = []
    if precautions:
        sections.append({"heading": "What You Can Do", "points": precautions})
    else:
        sections.append({
            "heading": "What You Can Do",
            "points": ["Consult a doctor for proper diagnosis."],
        })

    if urgency == "high":
        sections.append({
            "heading": "Urgent Action Required",
            "points": [
                "Seek medical attention immediately.",
                "Do not delay if symptoms are severe or worsening.",
            ],
        })
    elif urgency == "medium":
        sections.append({
            "heading": "Monitor Closely",
            "points": [
                "Watch symptoms over the next 24-48 hours.",
                "See a doctor if symptoms worsen or do not improve.",
            ],
        })
    else:
        sections.append({
            "heading": "General Advice",
            "points": [
                "Rest and stay hydrated.",
                "Consult a doctor for confirmation if symptoms persist.",
            ],
        })

    # summary — no mention of ML/pattern/model
    summary = ""

    return _base(
        "disease",
        disease,
        summary,                        # ← blank; was "Possible condition detected from your symptom pattern."
        sections=sections,
        confidence=confidence,
        matched_symptoms=matched,
        alternatives=alternatives,
        urgency=urgency,
        source=result.get("source", "symptom checker"),
    )


def build_pattern_response(result: dict) -> dict:
    """Low-confidence multi-candidate response — no ML wording."""
    disease    = result.get("disease", "Possible condition")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    matched    = result.get("matched_symptoms", [])
    if not isinstance(matched, list):
        matched = []

    ranked = [{"disease": disease, "confidence": confidence}]
    alts   = result.get("alternatives", [])
    if isinstance(alts, list):
        ranked.extend([a for a in alts if isinstance(a, dict)])

    ranked = [r for r in ranked if r.get("disease")]
    ranked = sorted(ranked, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[:3]

    points = [
        f"{item.get('disease')}: {round(float(item.get('confidence', 0.0)) * 100, 1)}%"
        for item in ranked
    ]
    points.append("Add symptom duration and one more symptom for a stronger result.")

    return _base(
        "disease",
        "Preliminary Analysis",            # ← was "Preliminary ML Pattern"
        "I found a low-confidence symptom pattern. Here are the closest matches.",  # no "ML"
        sections=[{"heading": "Closest matches", "points": points}],
        confidence=confidence,
        matched_symptoms=matched,
        alternatives=ranked[1:],
        urgency=result.get("urgency", "normal"),
        source="symptom checker",
    )


def build_error_response(message: str) -> dict:
    return _base("error", "Something went wrong", message, source="server")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _first_sentence(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = " ".join(sentences[:2])
    return out[:max_len] if len(out) > max_len else out


def _structure_answer(text: str) -> list[dict]:
    if not text or len(text) < 80:
        return [{"heading": "Answer", "points": [text]}] if text else []
    text      = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
    if len(sentences) <= 2:
        return [{"heading": "Key Information", "points": sentences}]
    chunk_size = 3
    headings   = ["Overview", "Key Details", "Additional Information", "More Information"]
    sections   = []
    for i, heading in enumerate(headings):
        chunk = sentences[i * chunk_size: i * chunk_size + chunk_size]
        if not chunk:
            break
        sections.append({"heading": heading, "points": [_trim(s) for s in chunk]})
    return sections


def _trim(s: str, max_words: int = 25) -> str:
    words = s.split()
    if len(words) <= max_words:
        return s
    return " ".join(words[:max_words]) + "..."

def build_model_not_ready_response(symptoms: list[str]) -> dict:
    """Shown when user reports symptoms but ML model has not been trained yet."""
    sym_str = ", ".join(symptoms[:5]) if symptoms else "your symptoms"
    return _base(
        "unknown",
        "Disease Model Not Ready",
        f"I detected your symptoms ({sym_str}), but the disease prediction model hasn't been trained yet.",
        sections=[{
            "heading": "How to enable disease prediction",
            "points": [
                "Step 1: Place 'Final_Augmented_dataset_Diseases_and_Symptoms.csv' in the data/ folder "
                "(rename it to 'symptoms_diseases.csv').",
                "Step 2: Place 'medquad.csv' in the data/ folder.",
                "Step 3: Run: python train_model.py",
                "Step 4: Restart the app: python app.py",
                "Training takes 3-8 minutes. After that, disease prediction will work for any symptoms you describe.",
            ],
        }],
        urgency="low",
        matched_symptoms=symptoms,
        source="system",
    )
