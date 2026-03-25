from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


ILLEGAL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def repair_mojibake(text: str) -> str:
    if not text:
        return text
    if not any(token in text for token in ("Ã", "Â", "â", "ï¿½")):
        return text
    try:
        repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        if repaired:
            return repaired
    except Exception:
        pass
    return text


def strip_boilerplate(text: str) -> str:
    patterns = [
        r"https?://sei\.inpi\.gov\.br/\S+",
        r"http://sei\.inpi\.gov\.br/\S+",
        r"sei/controlador\.php\?acao=documento[^\s]*",
        r"A autenticidade d[ea]ste documento pode ser conferida[^\n]*",
        r"Documento assinado eletronicamente[^\n]*",
        r"assinatura eletr[oô]nica[^\n]*",
        r"Expedido em \d{2}/\d{2}/\d{4}[^\n]*",
        r"MINIST[ÉE]RIO DA ECONOMIA",
        r"INSTITUTO NACIONAL DA PROPRIEDADE INDUSTRIAL",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_document_text(text: str) -> str:
    text = str(text or "").replace("_", " ")
    text = ILLEGAL_RE.sub("", text)
    text = repair_mojibake(text)
    text = strip_boilerplate(text)
    text = re.sub(r"(?:\b\d+\.\s*){10,}", " ", text)
    text = re.sub(r"(?:\b\d+\s*){20,}", " ", text)
    text = re.sub(r"[.·]{4,}\s*\d{1,3}", " ", text)
    text = re.sub(r"\b\d{1,3}\s+[A-Za-zÀ-ÿ]{2,}\s+\|\s+\d{1,2}/[A-Za-zÀ-ÿ]{3}/\d{4}\b", " ", text)
    text = re.sub(r"\b\d{1,3}\b(?=\s+[.·]{3,})", " ", text)
    text = re.sub(r"(?:\b[A-Z]{1,3}\b\s*){6,}", " ", text)
    return normalize_spaces(text)


def normalize_text(text: str) -> str:
    text = normalize_spaces(text).lower()
    return re.sub(r"[^\wÀ-ÿ\s/-]", " ", text)


def slugify_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return ascii_text or "documento"


def infer_label_from_filename(path: Path) -> str:
    return normalize_spaces(path.stem.replace("_", " ").replace("-", " "))[:140]


def infer_label_from_text(text: str, fallback: str) -> str:
    head = normalize_spaces(text)[:1200]
    patterns = [
        r"(DECRETO\s+N[ºO°]?\s*\d+[.,]?\d*)",
        r"(LEI\s+N[ºO°]?\s*\d+[.,]?\d*)",
        r"(PORTARIA[^\n]{0,80}N[ºO°]?\s*\d+[.,]?\d*)",
        r"(RESOLUÇÃO[^\n]{0,80}N[ºO°]?\s*\d+[.,]?\d*)",
        r"(INSTRUÇÃO\s+NORMATIVA[^\n]{0,80}N[ºO°]?\s*\d+[.,]?\d*)",
        r"(RELATÓRIO\s+ANUAL[^\n]{0,100})",
    ]
    for pattern in patterns:
        match = re.search(pattern, head, flags=re.IGNORECASE)
        if match:
            return normalize_spaces(match.group(1))
    first_sentence = re.split(r"(?<=[.!?])\s+", head)[0]
    return normalize_spaces(first_sentence)[:140] or fallback


def chunk_words(text: str, chunk_size: int = 220, overlap: int = 40) -> list[dict[str, int | str]]:
    words = clean_document_text(text).split()
    if not words:
        return []
    chunks: list[dict[str, int | str]] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if len(window) < 40 and start != 0:
            continue
        chunks.append(
            {
                "text": " ".join(window),
                "start_word": start,
                "end_word": start + len(window),
            }
        )
    return chunks


def read_text_file_robust(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.strip():
        return text
    return path.read_text(encoding="latin1", errors="ignore")


def load_json_documents(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("O JSON precisa estar no formato doc_id -> texto.")
    return {str(doc_id): clean_document_text(text) for doc_id, text in data.items()}


def normalized_name(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()


def find_documents_dir(base_path: Path, preferred_name: str = "Legislação e Atos Normativos Internos INPI") -> Path | None:
    direct = base_path / preferred_name
    if direct.exists():
        return direct

    wanted = normalized_name(preferred_name)
    for child in base_path.iterdir():
        if not child.is_dir():
            continue
        child_name = normalized_name(child.name)
        if child_name == wanted:
            return child
        if "atos normativos" in child_name and "inpi" in child_name:
            return child
    return None
