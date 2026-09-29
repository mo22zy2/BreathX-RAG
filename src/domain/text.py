"""Shared text/numeric helpers for the RAG answer pipeline.

Pure functions (no clients, no I/O) extracted from NLPController so the
safety classifier, confidence scorer, and citation verifier can share them
without duplicating logic.
"""
import re


def meaningful_tokens(text: str):
    stopwords = {
        "about", "after", "also", "based", "before", "being", "between",
        "could", "every", "from", "have", "into", "more", "must", "only",
        "should", "that", "their", "there", "these", "this", "those",
        "when", "where", "which", "with", "within", "would", "your",
        "document", "documents", "guideline", "guidelines", "source",
    }
    arabic_stopwords = {
        "وال", "في", "من", "إلى", "الى", "على", "عن", "مع", "هذا", "هذه",
        "ذلك", "التي", "الذي", "أن", "إن", "ما", "لم", "لن", "هل", "كل",
        "بعض", "عند", "لدى", "بعد", "قبل", "حتى", "أو", "ولا", "لكن",
        "كان", "كانت", "هي", "هو", "هم", "هن", "كما", "قد", "ثم", "حيث",
        "هناك", "إذا", "اذا", "ليس", "ليست", "أثناء", "خلال", "بين",
    }
    raw = text or ""
    # Strip Arabic diacritics/harakat so inflected forms still match.
    raw = re.sub(r"[\u064B-\u065F\u0670]", "", raw)
    lowered = raw.lower()

    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", lowered)
    # Arabic word tokens (Arabic script), including hamza variants.
    for word in re.findall(r"[\u0621-\u064A\u0671-\u06D3][\u0621-\u064A\u0671-\u06D3\-]*", raw):
        word = word.strip("-")
        if len(word) >= 2 and word not in arabic_stopwords:
            tokens.append(word)

    return [token for token in tokens if token not in stopwords]


def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def numeric_values(text: str):
    """Normalize numbers so '90%'==='90 %'==='90%' and '2.5'!=='25'."""
    normalized = []
    for token in re.findall(r"\d[\d,]*\.?\d*\s*%?|\d+", (text or "").replace(",", "")):
        token = token.replace(" ", "")
        if token not in normalized:
            normalized.append(token)
    return set(normalized)


def is_numeric_or_dosing_query(question: str):
    q = (question or "").strip()
    if not q:
        return False
    unit = re.compile(
        r"\b(mg|mcg|ug|ml|mcg/day|mg/day|times\s+a\s+day|twice\s+a\s+day|"
        r"once\s+a\s+day|puffs?|tablets?|doses?|dosage|dose|frequency|"
        r"dosages?|per\s+day|per\s+week|daily)\b",
        re.IGNORECASE,
    )
    quantity = re.compile(
        r"\b(how\s+many|how\s+much|how\s+often|what\s+dose|what\s+dosage|"
        r"dose\s+of|dosage\s+of)\b",
        re.IGNORECASE,
    )
    has_unit = bool(unit.search(q))
    has_digit = bool(re.search(r"\d", q))
    has_quantity = bool(quantity.search(q))
    return has_unit and (has_digit or has_quantity)


def normalize_section(text: str):
    normalized = re.sub(
        r"^(section|sec|heading)\s*[:.#-]?\s*",
        "",
        (text or "").lower().strip(),
    )
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .,;:-()[]\t")


def has_official_evidence_metadata(doc: dict):
    metadata = doc.get("metadata") or {}
    document_name = (metadata.get("document_name") or "").lower()
    org = (metadata.get("org") or "").lower()
    source_url = (metadata.get("source_url") or "").lower()
    page_number = metadata.get("page_number")
    official_markers = ("nice", "gina", "nhlbi", "naepp", "who", "cdc", "uspstf")
    _official_re = re.compile(r'(?:^|[\s_/]|\.)(?:' + '|'.join(re.escape(m) for m in official_markers) + r')(?:[\s_/]|\.|$)', re.IGNORECASE)
    combined = " ".join([document_name, org, source_url])
    has_source = bool(_official_re.search(combined))
    return has_source and page_number not in (None, "", 0)
