"""
symptom_extractor.py — Expanded keyword mapping for maximum symptom coverage.
Maps user-typed phrases to CSV column names used by the ML model.
"""
from __future__ import annotations
import re
from utils.medical_knowledge import COMMON_SYMPTOMS, SYMPTOM_ALIASES, dedupe_preserve_order, normalize_text, tokenize

KNOWN_SYMPTOMS: list[str] = []

def load_symptom_list(symptom_list: list[str] = None):
    """Set the master symptom list (usually from CSV or fallback)."""
    global KNOWN_SYMPTOMS
    if symptom_list and len(symptom_list) > 50:
        KNOWN_SYMPTOMS = dedupe_preserve_order(symptom_list)
    else:
        KNOWN_SYMPTOMS = list(COMMON_SYMPTOMS)

def _match_phrase(text: str, phrase: str) -> bool:
    """Case‑insensitive whole‑word matching with spaces allowed."""
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return bool(re.search(rf"\b{escaped}\b", text, re.IGNORECASE))

# ── Body part → symptom mapping ──
BODY_PAIN_MAP = {
    "hand": "hand pain", "hands": "hand pain", "wrist": "wrist pain", "wrists": "wrist pain",
    "finger": "hand pain", "fingers": "hand pain", "leg": "leg pain", "legs": "leg pain",
    "thigh": "leg pain", "calf": "leg pain", "ankle": "leg pain", "foot": "foot pain",
    "feet": "foot pain", "toe": "foot pain", "knee": "knee pain", "knees": "knee pain",
    "head": "headache", "stomach": "stomach pain", "abdomen": "abdominal pain",
    "chest": "chest pain", "heart": "chest pain", "throat": "sore throat",
    "ear": "ear pain", "ears": "ear pain",
    "eyes": "eye pain", "eye": "eye pain",
    "joint": "joint pain", "joints": "joint pain",
    "muscle": "muscle pain", "muscles": "muscle pain",
    "back": "back pain", "neck": "neck pain", "shoulder": "shoulder pain",
    "arm": "arm pain", "elbow": "arm pain",
    "whole body": "body aches", "body": "body aches",
    "hip": "hip pain", "hips": "hip pain",
    "pelvis": "pelvic pain", "pelvic": "pelvic pain",
    "face": "facial pain", "jaw": "jaw pain",
}

# ── Extended keyword map — covers common phrases, slang, and medical terms ──
KEYWORD_MAP = {
    # ── Core symptoms ──
    r"\bfever\b": "fever",
    r"\bhigh temperature\b": "fever",
    r"\btemperature\b": "fever",
    r"\bcough\b": "cough",
    r"\bdry cough\b": "cough",
    r"\bwet cough\b": "cough",
    r"\bheadache\b": "headache",
    r"\bhead\s*ache\b": "headache",
    r"\bnausea\b": "nausea",
    r"\bqueasy\b": "nausea",
    r"\bfeel(ing)?\s+sick\b": "nausea",
    r"\bvomit(ing|ed)?\b": "vomiting",
    r"\bthrow(ing)?\s+up\b": "vomiting",
    r"\bpuking\b": "vomiting",
    r"\bdiarrhea\b": "diarrhea",
    r"\bdiarrhoea\b": "diarrhea",
    r"\bloosemotions?\b": "diarrhea",
    r"\bloose\s*motions?\b": "diarrhea",
    r"\bwatery\s+stool\b": "diarrhea",
    r"\bloose\s+stool\b": "diarrhea",
    r"\bconstipation\b": "constipation",
    r"\bconstipated\b": "constipation",
    r"\bfatigue\b": "fatigue",
    r"\btired(ness)?\b": "fatigue",
    r"\bexhausted\b": "fatigue",
    r"\bexhaustion\b": "fatigue",
    r"\bweakness\b": "weakness",
    r"\bweak\b": "weakness",
    r"\bdizziness\b": "dizziness",
    r"\bdizzy\b": "dizziness",
    r"\bvertigo\b": "dizziness",
    r"\blightheaded\b": "dizziness",
    r"\brash\b": "rash",
    r"\bskin\s+rash\b": "rash",
    r"\bhives\b": "rash",
    r"\bitching\b": "itching",
    r"\bitchy\b": "itching",
    r"\bpruritus\b": "itching",
    r"\bchills\b": "chills",
    r"\bsweating\b": "sweating",
    r"\bnight\s+sweat(s)?\b": "sweating",
    r"\bnosebleed\b": "nosebleed",
    r"\bnose\s*bleed\b": "nosebleed",
    r"\bepistaxis\b": "nosebleed",

    # ── Respiratory ──
    r"\bshortness\s+of\s+breath\b": "shortness of breath",
    r"\bbreath(ing)?\s+difficult\w*\b": "shortness of breath",
    r"\bdifficulty\s+breath(ing)?\b": "shortness of breath",
    r"\bbreathless(ness)?\b": "shortness of breath",
    r"\bwheezing\b": "wheezing",
    r"\bwheeze\b": "wheezing",

    # ── Cardiac ──
    r"\bpalpitations?\b": "palpitations",
    r"\bheart\s+rac(ing|e)\b": "palpitations",
    r"\bheart\s+pounding\b": "palpitations",
    r"\birregular\s+heartbeat\b": "palpitations",
    r"\bchest\s+pain\b": "chest pain",
    r"\bchest\s+tightness\b": "chest pain",
    r"\bchest\s+pressure\b": "chest pain",
    r"\bchest\s+discomfort\b": "chest pain",
    r"\bheart\s+pain\b": "chest pain",
    r"\bpain\s+in\s+(my\s+)?heart\b": "chest pain",
    r"\bchec?t\s+pain\b": "chest pain",  # typo rescue

    # ── Neurological ──
    r"\bnumbness\b": "numbness",
    r"\bnumb\b": "numbness",
    r"\btingling\b": "tingling",
    r"\bpins\s+and\s+needles\b": "tingling",
    r"\bblurred?\s+vision\b": "blurred vision",
    r"\bdifficulty\s+speak(ing)?\b": "difficulty speaking",
    r"\bslurred\s+speech\b": "difficulty speaking",
    r"\bseizure\b": "seizure",
    r"\bconvulsion\b": "seizure",
    r"\btrembling\b": "tremor",
    r"\btremor\b": "tremor",
    r"\bshaking\b": "tremor",
    r"\bshaky\b": "tremor",
    r"\bconfusion\b": "confusion",
    r"\bconfused\b": "confusion",
    r"\bmemory\s+loss\b": "memory loss",
    r"\bforgetful\b": "memory loss",
    r"\bdifficulty\s+walk(ing)?\b": "difficulty walking",

    # ── Skin & eye ──
    r"\bjaundice\b": "jaundice",
    r"\byellow\s+skin\b": "jaundice",
    r"\byellow\s+eyes\b": "jaundice",
    r"\byellowing\b": "jaundice",
    r"\bskin\s+yellowing\b": "jaundice",
    r"\bhair\s+loss\b": "hair loss",
    r"\balopecia\b": "hair loss",
    r"\bbald(ness|ing)?\b": "hair loss",
    r"\bswollen\s+(lymph\s+)?nodes?\b": "swollen lymph nodes",
    r"\blymph\s+node\s+swelling\b": "swollen lymph nodes",
    r"\bglands?\s+swollen\b": "swollen lymph nodes",
    r"\bneck\s+glands?\b": "swollen lymph nodes",
    r"\bswollen\s+neck\b": "swollen lymph nodes",

    # ── Musculoskeletal ──
    r"\bjoint\s+pain\b": "joint pain",
    r"\baching\s+joints?\b": "joint pain",
    r"\bjoint\s+ache\b": "joint pain",
    r"\bstiff(ness)?\b": "stiffness",
    r"\bjoint\s+stiffness\b": "stiffness",
    r"\bmorning\s+stiffness\b": "stiffness",
    r"\bswelling\b": "swelling",
    r"\bswollen\b": "swelling",
    r"\binflammation\b": "swelling",
    r"\bpuffy\b": "swelling",
    r"\bmuscle\s+pain\b": "muscle pain",
    r"\bmyalgia\b": "muscle pain",
    r"\bbody\s+aches?\b": "body aches",
    r"\bache\s+all\s+over\b": "body aches",
    r"\bback\s+pain\b": "back pain",
    r"\blower\s+back\s+pain\b": "lower back pain",
    r"\blumbar\s+pain\b": "lower back pain",

    # ── GI ──
    r"\babdominal\s+pain\b": "abdominal pain",
    r"\bstomach\s+ache\b": "abdominal pain",
    r"\bbloat(ing|ed)?\b": "bloating",
    r"\bgas\b": "bloating",
    r"\bmucus\s+in\s+stool\b": "mucus in stool",
    r"\bblood\s+in\s+stool\b": "blood in stool",
    r"\bmelena\b": "blood in stool",
    r"\bappetite\s+loss\b": "loss of appetite",
    r"\bloss\s+of\s+appetite\b": "loss of appetite",
    r"\bno\s+appetite\b": "loss of appetite",
    r"\bnot\s+hungry\b": "loss of appetite",

    # ── Weight / metabolic ──
    r"\bweight\s+loss\b": "weight loss",
    r"\blosing\s+weight\b": "weight loss",
    r"\bunintended\s+weight\s+loss\b": "weight loss",
    r"\bweight\s+gain\b": "weight gain",
    r"\bgaining\s+weight\b": "weight gain",
    r"\bexcessive\s+thirst\b": "excessive thirst",
    r"\bpolydipsia\b": "excessive thirst",
    r"\bincreased\s+thirst\b": "excessive thirst",
    r"\bthirst\b": "excessive thirst",
    r"\bfrequent\s+urinat(ion|ing)\b": "frequent urination",
    r"\bpolyuria\b": "frequent urination",
    r"\bpee(ing)?\s+(a\s+lot|frequently)\b": "frequent urination",
    r"\bpainful\s+urinat(ion|ing)\b": "painful urination",
    r"\bdysuria\b": "painful urination",
    r"\bburning\s+(when\s+)?(pee|urinat)\w*\b": "painful urination",
    r"\bheat\s+intolerance\b": "heat intolerance",
    r"\bcold\s+intolerance\b": "cold intolerance",
    r"\bsensitive\s+to\s+cold\b": "cold intolerance",

    # ── Sleep / mood ──
    r"\binsomnia\b": "insomnia",
    r"\btrouble\s+sleep(ing)?\b": "insomnia",
    r"\bcannot?\s+sleep\b": "insomnia",
    r"\banxiety\b": "anxiety",
    r"\bpanic\b": "anxiety",
    r"\bdepression\b": "depression",
    r"\bdepressed\b": "depression",
    r"\bmood\s+swings?\b": "mood swings",

    # ── ENT ──
    r"\bsore\s+throat\b": "sore throat",
    r"\bthroat\s+pain\b": "sore throat",
    r"\bdifficulty\s+swallow(ing)?\b": "difficulty swallowing",
    r"\bdysphagia\b": "difficulty swallowing",
    r"\bpainful\s+swallow(ing)?\b": "difficulty swallowing",
    r"\bsneezing\b": "sneezing",
    r"\bsneezes?\b": "sneezing",
    r"\brunny\s+nose\b": "runny nose",
    r"\bnasal\s+discharge\b": "runny nose",
    r"\bstuffy\s+nose\b": "nasal congestion",
    r"\bnasal\s+congestion\b": "nasal congestion",
    r"\bblocked\s+nose\b": "nasal congestion",
    r"\bear\s+pain\b": "ear pain",
    r"\bearache\b": "ear pain",
    r"\bpain\s+in\s+(my\s+)?ears?\b": "ear pain",
    r"\bhearing\s+loss\b": "hearing loss",
    r"\bdifficulty\s+hear(ing)?\b": "hearing loss",
    r"\bhard\s+of\s+hearing\b": "hearing loss",
    r"\bdischarge\s+from\s+ear\b": "ear discharge",
    r"\bear\s+discharge\b": "ear discharge",
    r"\bear\s+drainage\b": "ear discharge",
    r"\bwatery\s+eyes?\b": "watery eyes",
    r"\beyes?\s+water(ing)?\b": "watery eyes",
    r"\blight\s+sensitivit\w*\b": "light sensitivity",
    r"\bphotophobia\b": "light sensitivity",
    r"\bsensitive\s+to\s+light\b": "light sensitivity",

    # ── Menstrual / female ──
    r"\bperiods?\b": "painful periods",
    r"\bperiod\s+pain\b": "painful periods",
    r"\bmenstrual\s+(pain|cramps?|ache)\b": "painful periods",
    r"\bmenstruation\b": "painful periods",
    r"\bdysmenorrhea\b": "painful periods",
    r"\birregular\s+periods?\b": "irregular periods",
    r"\bheavy\s+periods?\b": "heavy periods",
    r"\bvaginal\s+discharge\b": "vaginal discharge",
    r"\bpelvic\s+pain\b": "pelvic pain",
    r"\bpcos\b": "irregular periods",

    # ── Urinary ──
    r"\bblood\s+in\s+urine\b": "blood in urine",
    r"\bhematuria\b": "blood in urine",

    # ── General / vague ──
    r"\bnot\s+feeling\s+well\b": "fatigue",
    r"\bunwell\b": "fatigue",
    r"\bill\b": "fatigue",
    r"\bmalaise\b": "fatigue",
    r"\blethargy\b": "fatigue",
    r"\blethargic\b": "fatigue",

    # ── Skin changes ──
    r"\bpale\s+skin\b": "pallor",
    r"\bpallor\b": "pallor",
    r"\bskin\s+pale\b": "pallor",
    r"\bnail\s+changes?\b": "nail changes",
    r"\bbrittle\s+nails?\b": "nail changes",
}

def _extract_by_aliases(text: str) -> list[str]:
    """Use SYMPTOM_ALIASES (from medical_knowledge) for common variants."""
    found = []
    for alias in sorted(SYMPTOM_ALIASES.keys(), key=len, reverse=True):
        if _match_phrase(text, alias):
            found.append(SYMPTOM_ALIASES[alias])
    return found

def _extract_by_known_symptoms(text: str) -> list[str]:
    """Match against the loaded symptom list (from CSV or built-in)."""
    if not KNOWN_SYMPTOMS:
        return []
    found = []
    multi = sorted([s for s in KNOWN_SYMPTOMS if " " in s], key=len, reverse=True)
    for symptom in multi:
        if _match_phrase(text, symptom):
            found.append(symptom)
    tokens = set(tokenize(text))
    single = [s for s in KNOWN_SYMPTOMS if " " not in s]
    for symptom in single:
        if symptom in tokens:
            found.append(symptom)
    return found

def _extract_body_area_pain(text: str) -> list[str]:
    """Detect '<body part> pain', '<body part> hurts', etc."""
    found = []
    for part, symptom in BODY_PAIN_MAP.items():
        patterns = [
            rf"\b{part}\s+(pain|ache|hurts?)\b",
            rf"\b(hurting|aching)\s+{part}\b",
            rf"\bpain\s+in\s+(my\s+)?{part}\b",
            rf"\bswollen\s+{part}\b",
        ]
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.append(symptom)
                break
    return found

def _extract_keyword_map(text: str) -> list[str]:
    """Apply the regex keyword map (covers idioms and slang)."""
    found = []
    for pattern, symptom in KEYWORD_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(symptom)
    return found

def extract_symptoms(user_text: str) -> list[str]:
    """Main entry point: return a deduplicated list of normalised symptoms."""
    if not user_text:
        return []
    text = normalize_text(user_text)
    found = []
    found.extend(_extract_by_aliases(text))
    found.extend(_extract_by_known_symptoms(text))
    found.extend(_extract_body_area_pain(text))
    found.extend(_extract_keyword_map(text))

    # If generic "body aches" is present but more specific pains exist, keep specifics only
    specific = [s for s in found if s != "body aches"]
    if specific:
        found = specific

    # Common misspelling rescue
    if "chest pain" not in found and re.search(r"\bchec?t\s+pain\b", text, re.IGNORECASE):
        found.append("chest pain")

    return dedupe_preserve_order(found)