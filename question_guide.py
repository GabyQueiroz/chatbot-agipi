from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class QuestionExample:
    question: str
    category: str
    document: str


@dataclass
class QuestionMatch:
    question: str
    category: str
    document: str
    score: float


class QuestionGuide:
    def __init__(self, examples: list[QuestionExample]) -> None:
        self.examples = examples
        texts = [self._compose_text(item.question, item.category) for item in examples]

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)

    @classmethod
    def from_excel(cls, path: Path) -> "QuestionGuide":
        df = pd.read_excel(path)
        required = {"Pergunta"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes em {path.name}: {sorted(missing)}")

        examples: list[QuestionExample] = []
        for _, row in df.fillna("").iterrows():
            question = str(row.get("Pergunta", "")).strip()
            if not question:
                continue
            examples.append(
                QuestionExample(
                    question=question,
                    category=str(row.get("Categoria", "")).strip(),
                    document=str(row.get("Documento", "")).strip(),
                )
            )
        if not examples:
            raise ValueError("Nenhuma pergunta válida foi encontrada na planilha.")
        return cls(examples)

    def match(self, question: str, top_k: int = 5) -> list[QuestionMatch]:
        query_text = self._compose_text(question, "")
        q_word = self.word_vectorizer.transform([query_text])
        q_char = self.char_vectorizer.transform([query_text])
        word_scores = (self.word_matrix @ q_word.T).toarray().ravel()
        char_scores = (self.char_matrix @ q_char.T).toarray().ravel()
        scores = (0.7 * word_scores) + (0.3 * char_scores)

        if len(scores) == 0:
            return []

        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        matches: list[QuestionMatch] = []
        for idx in top_idx:
            example = self.examples[int(idx)]
            matches.append(
                QuestionMatch(
                    question=example.question,
                    category=example.category,
                    document=example.document,
                    score=float(scores[idx]),
                )
            )
        return matches

    def infer_category(self, question: str, min_score: float = 0.18) -> str | None:
        matches = self.match(question, top_k=3)
        if not matches:
            return None
        best = matches[0]
        if best.score < min_score or not best.category:
            return None
        return best.category

    def build_retrieval_hint(self, question: str) -> str:
        matches = self.match(question, top_k=3)
        if not matches:
            return question

        best = matches[0]
        lines = [question]
        if best.category and best.score >= 0.18:
            lines.append(f"Categoria provável: {best.category}")
        return "\n".join(lines)

    @staticmethod
    def _compose_text(question: str, category: str) -> str:
        category = category or ""
        if category:
            return f"{question} {category} {category}"
        return question
