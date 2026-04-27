from __future__ import annotations
import re
from utils.medical_knowledge import normalize_text

# UI chip texts — don't process as real queries
SUGGESTION_CHIP_TEXTS = {
    "ask about symptoms",
    "describe your age and duration",
    "tell me which body area is affected",
    "add more symptoms",
    "tell me how long this has been going on",
    "ask when to seek urgent care",
}

GREETING_EXACT = {
    "hi", "hello", "hey", "hii", "helo", "hai", "hiya", "howdy", "hie",
    "good morning", "good afternoon", "good evening", "good night",
    "namaste", "namaskar", "salaam",
}

GREETING_PATTERNS = [
    r"^hi+[!.\s]*$",
    r"^hey+[!.\s]*$",
    r"^hello+[!.\s]*$",
    r"^(hi|hey|hello)\s+there[!.\s]*$",
    r"^good\s+(morning|afternoon|evening|night)[!.\s]*$",
]

NON_MEDICAL_EXACT = {
    "ok", "okay", "k", "sure", "alright", "fine",
    "thanks", "thank you", "thank u", "thx", "ty",
    "bye", "goodbye", "see you", "see ya",
    "great", "nice", "wow", "cool", "awesome", "got it",
    "lol", "haha", "hehe",
}

NON_MEDICAL_PATTERNS = [
    r"^how are you[?.\s]*$",
    r"^how r u[?.\s]*$",
    r"^how are u[?.\s]*$",
    r"^(hi|hey|hello)[,\s]+how are you[?.\s]*$",
    r"^(hi|hey|hello)[,\s]+how r u[?.\s]*$",
    r"^who are you[?.\s]*$",
    r"^what are you[?.\s]*$",
    r"^what is your name[?.\s]*$",
    r"^are you (a bot|ai|a robot|real|human)[?.\s]*$",
    r"^who (made you|created you|built you)[?.\s]*$",
    r"\b(cricket|football|soccer|movie|song|music|joke|funny|weather|politics|news)\b",
]

# Strong question starters
QUESTION_PATTERNS = [
    r"\bwhat is\b", r"\bwhat are\b", r"\bwhat causes\b",
    r"\bwhat should i (do|eat|avoid|take)\b",
    r"\bhow to (treat|manage|cure|reduce|prevent|handle|stop)\b",
    r"\bhow (do i|can i|should i)\b",
    r"\bwhy (do|does|is|am)\b",
    r"\bwhen should i\b", r"\bwhen to\b",
    r"\bis it (serious|dangerous|contagious|normal|safe)\b",
    r"\bshould i (see|go|take|use)\b",
    r"\btell me about\b", r"\bexplain\b", r"\bdefine\b",
    r"\btreatment for\b", r"\bremedy for\b", r"\bhome remedies?\b",
    r"\bsymptoms of\b", r"\bsigns of\b",
]

# These single words are medical symptoms — even alone they are symptom intent
SINGLE_WORD_SYMPTOMS = {
    "fever", "cough", "headache", "nausea", "vomiting", "diarrhea",
    "fatigue", "weakness", "dizziness", "rash", "itching", "chills",
    "constipation", "wheezing", "jaundice", "insomnia", "numbness",
    "tingling", "sweating", "sneezing", "bleeding", "fainting",
}

SYMPTOM_KEYWORDS = [
    r"\b(fever|headache|cough|cold|vomiting|nausea|diarrhea|rash|chills|sneez)\b",
    r"\b(fatigue|weakness|dizziness|swelling|itching|burning|bleeding)\b",
    r"\b(sore throat|runny nose|blocked nose|chest pain|back pain)\b",
    r"\b(stomach pain|leg pain|arm pain|knee pain|neck pain|joint pain)\b",
    r"\b(loose motion|loosemotion|loose stool|watery stool|constipation|loosemotions?)\b",
    r"\b(shortness of breath|difficulty breathing|breathlessness)\b",
    r"\b(period pain|period cramps|menstrual pain|dysmenorrhea|periods?)\b",
    r"\b(palpitation|heart racing|rapid heartbeat|irregular heartbeat)\b",
    r"\b(jaundice|yellow eye|yellow skin)\b",
    r"\b(hair loss|weight loss|weight gain|loss of appetite)\b",
    r"\b(insomnia|blurred vision|numbness|tingling)\b",
    r"\b(vomitting|vomittingh|loosemotion)\b",  # common typos
]

REPORTING_TRIGGERS = [
    r"\bi (have|had|got|get)\b",
    r"\bi (am|was) (having|experiencing|feeling|suffering)\b",
    r"\bi feel\b", r"\bi'm feeling\b",
    r"\bi (am|feel) (sick|ill|unwell|bad)\b",
    r"\bnot feeling (well|good)\b",
    r"\bsuffering from\b",
    r"\bpain in\b",
    r"\b\w+ pain\b",
]

VAGUE_UNWELL = [
    r"\bnot feeling well\b", r"\bnot feeling good\b",
    r"\bfeeling (sick|ill|unwell|bad|terrible|awful)\b",
    r"\bnot (okay|ok|well|good)\b",
    r"\b(sick|ill|unwell)\b",
]


def detect_intent(message: str, has_symptoms: bool) -> str:
    msg = normalize_text(message)
    if not msg:
        return "unknown"

    if msg in SUGGESTION_CHIP_TEXTS:
        return "non_medical"

    # Greeting
    if msg in GREETING_EXACT:
        return "greeting"
    if any(re.match(p, msg) for p in GREETING_PATTERNS):
        return "greeting"

    # Pure non-medical
    if msg in NON_MEDICAL_EXACT:
        return "non_medical"
    if any(re.match(p, msg) for p in NON_MEDICAL_PATTERNS):
        return "non_medical"

    # Single-word check — if it's a known symptom word, it's a symptom report
    if msg in SINGLE_WORD_SYMPTOMS:
        return "symptom"

    # has_symptoms flag set by extractor
    if has_symptoms:
        # But if phrased as a question about a disease, it's a question
        info_starters = [
            r"^tell me (about|regarding)\b",
            r"^explain\b", r"^define\b",
            r"^what (is|are|causes|happens)\b",
            r"^symptoms of\b", r"^signs of\b",
        ]
        if any(re.match(p, msg) for p in info_starters):
            return "question"
        # Advice question
        if any(re.search(p, msg) for p in [
            r"\bwhat should i do\b", r"\bwhat to do\b",
            r"\bhow to treat\b", r"\bhow to manage\b", r"\bhow can i\b",
        ]):
            return "question"
        return "symptom"

    is_vague = any(re.search(p, msg) for p in VAGUE_UNWELL)
    is_reporting = is_vague or any(re.search(p, msg) for p in REPORTING_TRIGGERS)
    has_sym_word = any(re.search(p, msg) for p in SYMPTOM_KEYWORDS)

    is_question = any(re.search(p, msg) for p in QUESTION_PATTERNS) or msg.endswith("?")

    # Social phrases without medical content
    social = ["how are you", "how r u", "what are you doing", "where are you", "who are you"]
    if any(s in msg for s in social) and not has_sym_word:
        return "non_medical"

    if has_sym_word and is_question:
        # "tell me about migraine" → question
        if any(re.match(p, msg) for p in [
            r"^tell me\b", r"^explain\b", r"^what (is|are|causes)\b",
        ]):
            return "question"
        # "what should i do for cough" → question
        if any(re.search(p, msg) for p in [
            r"\bwhat should i do\b", r"\bhow to\b", r"\bwhat to do\b",
        ]):
            return "question"
        return "symptom"

    if has_sym_word or is_reporting:
        return "symptom"

    if is_question:
        return "question"

    # Short message with no recognized content
    if len(msg.split()) <= 2:
        return "non_medical"

    return "unknown"