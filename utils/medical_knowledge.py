from __future__ import annotations

import re

DEFAULT_PROMPTS = [
    "What causes Sudden Cardiac Arrest ?",
    "What are the symptoms of Sudden Cardiac Arrest ?",
    "What causes Urinary Tract Infections ?",
    "How to prevent Glaucoma ?",
]

COMMON_SYMPTOMS = [
    "fever", "chills", "fatigue", "weakness", "dizziness", "headache", "migraine",
    "cough", "sore throat", "runny nose", "blocked nose", "congestion", "wheezing",
    "shortness of breath", "difficulty breathing", "chest pain", "chest tightness",
    "palpitations", "rapid heartbeat", "nausea", "vomiting", "diarrhea", "loose motions",
    "constipation", "abdominal pain", "stomach pain", "bloating", "heartburn",
    "burning urination", "frequent urination", "painful urination", "blood in urine",
    "back pain", "lower back pain", "flank pain", "rash", "itching", "swelling",
    "body aches", "muscle pain", "joint pain", "neck pain", "shoulder pain",
    "knee pain", "leg pain", "arm pain", "thirst", "dry mouth", "increased hunger",
    "blurred vision", "numbness", "tingling", "hair loss", "weight loss", "weight gain",
    "insomnia", "trouble sleeping", "loss of appetite", "jaundice", "yellow skin", "yellow eyes",
    "difficulty speaking",
]

SYMPTOM_ALIASES = {
    "temperature": "fever",
    "feverish": "fever",
    "loose motion": "diarrhea",
    "loose motions": "diarrhea",
    "loosemotion": "diarrhea",
    "loosemotions": "diarrhea",
    "watery stool": "diarrhea",
    "stomach ache": "abdominal pain",
    "tummy pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "throwing up": "vomiting",
    "vomitting": "vomiting",
    "vomittingh": "vomiting",
    "puking": "vomiting",
    "head pain": "headache",
    "breathlessness": "shortness of breath",
    "cant breathe": "shortness of breath",
    "trouble breathing": "shortness of breath",
    "blocked nose": "congestion",
    "stuffy nose": "congestion",
    "throat pain": "sore throat",
    "body ache": "body aches",
    "back ache": "back pain",
    "heart pain": "chest pain",
    "pain in heart": "chest pain",
    "heart racing": "palpitations",
    "always tired": "fatigue",
    "blurry vision": "blurred vision",
    "cannot sleep": "insomnia",
}


def normalize_text(text: str) -> str:
    lowered = str(text).lower().replace("/", " ")
    lowered = re.sub(r"[^a-z0-9\s']+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_text(text))


def dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
