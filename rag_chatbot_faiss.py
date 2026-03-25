from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from document_utils import normalize_text
from prepare_chatbot_corpus import build_chunks, build_documents, build_manifest, write_jsonl
from question_guide import QuestionExample, QuestionGuide, QuestionMatch

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


DEFAULT_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_USE_LLM = True
STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas", "de", "da", "das", "do", "dos", "e", "em", "no", "na",
    "nos", "nas", "para", "por", "com", "sem", "sobre", "que", "qual", "quais", "como", "tem", "ha", "algum",
    "alguma", "alguns", "algumas", "ser", "seja", "sao", "foi", "sido", "ao", "aos", "ou", "se", "sua", "suas",
    "seu", "seus", "uepg",
}


@dataclass
class FaissChunk:
    chunk_id: str
    doc_id: str
    original_doc_id: str
    corpus: str
    label: str
    source_type: str
    source_path: str
    chunk_index: int
    start_word: int
    end_word: int
    text: str


@dataclass
class RetrievedPassage:
    chunk: FaissChunk
    score: float


@dataclass
class FAQEntry:
    question: str
    category: str
    answer: str
    evidence: str
    source_docs: str


def normalize_spaces(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def torch_dtype_for_device() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def ensure_prepared_corpus(base_path: Path) -> Path:
    output_dir = base_path / "saida" / "prepared_corpus"
    manifest_path = output_dir / "manifest.json"
    chunks_path = output_dir / "chunks.jsonl"
    documents_path = output_dir / "documents.jsonl"
    if manifest_path.exists() and chunks_path.exists() and documents_path.exists():
        return output_dir

    documents = build_documents(base_path)
    if not documents:
        raise FileNotFoundError("Nenhum documento foi encontrado para montar a base preparada.")

    chunks = build_chunks(documents, chunk_size=220, overlap=40)
    manifest = build_manifest(documents, chunks, chunk_size=220, overlap=40)
    write_jsonl(documents_path, documents)
    write_jsonl(chunks_path, chunks)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def load_chunks(path: Path) -> list[FaissChunk]:
    chunks: list[FaissChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            chunks.append(FaissChunk(**raw))
    return chunks


def normalize_question_key(text: str) -> str:
    return normalize_spaces(str(text or "")).casefold()


def tokenize_for_search(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = [token for token in normalized.split() if len(token) >= 3 and token not in STOPWORDS]
    return tokens


def expand_query_terms(question: str) -> list[str]:
    terms = set(tokenize_for_search(question))
    normalized = normalize_text(question)

    if "software" in normalized or "programa de computador" in normalized or "programa" in terms:
        terms.update({"software", "programa", "computador", "registro"})
    if "registro" in normalized or "registr" in normalized:
        terms.update({"registro", "registrado", "cadastro"})
    if any(term in normalized for term in ("medicina", "medico", "medica", "clinica", "hospital", "saude")):
        terms.update({"medicina", "medico", "medica", "saude", "clinica", "hospital"})
    if "patente" in normalized:
        terms.update({"patente", "invento", "propriedade"})

    return sorted(terms)


def clean_user_facing_answer(text: str) -> str:
    cleaned = str(text or "")
    replacements = {
        "de acordo com o documento": "",
        "de acordo com os documentos": "",
        "com base no documento": "",
        "com base nos documentos": "",
        "nos trechos recuperados": "",
        "nos trechos analisados": "",
        "no documento": "",
        "nos documentos": "",
        "no trecho": "",
        "nos trechos": "",
        "em anexo": "",
        "no anexo": "",
        "nos anexos": "",
        "conforme anexo": "",
        "conforme o anexo": "",
        "conforme documento": "",
        "conforme o documento": "",
    }
    lowered = cleaned
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
        lowered = lowered.replace(source.capitalize(), target)
    lowered = lowered.replace("  ", " ")
    lowered = lowered.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
    lowered = lowered.replace("()", "")
    lowered = normalize_spaces(lowered)
    return lowered


class FAQAnswerBank:
    def __init__(self, entries: list[FAQEntry]) -> None:
        self.entries = entries
        self.exact_map = {normalize_question_key(entry.question): entry for entry in entries}
        self.guide = QuestionGuide(
            [QuestionExample(question=entry.question, category=entry.category, document=entry.source_docs) for entry in entries]
        )
        self.question_map = {entry.question: entry for entry in entries}

    @classmethod
    def from_excel(cls, path: Path) -> "FAQAnswerBank":
        df_raw = pd.read_excel(path, header=None)
        header_row = None
        for idx, row in df_raw.iterrows():
            first = normalize_spaces(row.iloc[0])
            second = normalize_spaces(row.iloc[1]) if len(row) > 1 else ""
            if first == "Pergunta" and second == "Categoria":
                header_row = idx
                break
        if header_row is None:
            raise ValueError(f"Não foi possível localizar o cabeçalho da FAQ em {path.name}.")

        headers = [normalize_spaces(value) for value in df_raw.iloc[header_row].tolist()]
        df = df_raw.iloc[header_row + 1 :].copy()
        df.columns = headers
        for column in ("Pergunta", "Categoria", "Resposta", "Evidência", "Base documental"):
            if column not in df.columns:
                df[column] = ""

        entries: list[FAQEntry] = []
        for _, row in df.fillna("").iterrows():
            question = normalize_spaces(row.get("Pergunta", ""))
            answer = normalize_spaces(row.get("Resposta", ""))
            if not question or not answer:
                continue
            entries.append(
                FAQEntry(
                    question=question,
                    category=normalize_spaces(row.get("Categoria", "")),
                    answer=answer,
                    evidence=normalize_spaces(row.get("Evidência", "")),
                    source_docs=normalize_spaces(row.get("Base documental", "")),
                )
            )
        if not entries:
            raise ValueError(f"Nenhuma entrada válida foi encontrada em {path.name}.")
        return cls(entries)

    def match(self, question: str) -> tuple[FAQEntry | None, float]:
        exact = self.exact_map.get(normalize_question_key(question))
        if exact is not None:
            return exact, 1.0

        heuristic = self._heuristic_match(question)
        if heuristic is not None:
            return heuristic, 0.9

        matches = self.guide.match(question, top_k=1)
        if not matches:
            return None, 0.0
        best = matches[0]
        entry = self.question_map.get(best.question)
        if entry is None:
            return None, 0.0
        return entry, float(best.score)

    def _heuristic_match(self, question: str) -> FAQEntry | None:
        normalized = normalize_question_key(question)
        definition_intents = (
            "o que e",
            "oque e",
            "explique",
            "me fale",
            "fale sobre",
            "defina",
            "conceito",
            "qual e",
        )
        if not any(intent in normalized for intent in definition_intents):
            return None

        priority_pairs = [
            ("ageuni", "o que é o programa ageuni?"),
            ("agipi", "o que é a agipi da uepg?"),
            ("epitec", "o que é o epitec dentro da agipi?"),
            ("inprotec", "o que é o inprotec dentro da agipi?"),
        ]
        for keyword, target_question in priority_pairs:
            if keyword in normalized:
                return self.exact_map.get(target_question)
        return None


class InstructionLLM:
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL) -> None:
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self) -> "InstructionLLM":
        if self.tokenizer is not None and self.model is not None:
            return self
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype_for_device(),
            device_map="auto",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self

    def generate(self, prompt: str, max_new_tokens: int = 420) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=7000)
        if hasattr(self.model, "device") and str(self.model.device) != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class FaissKnowledgeBase:
    def __init__(self, chunks_path: Path, index_dir: Path, embedding_model_name: str = DEFAULT_SBERT_MODEL) -> None:
        self.chunks_path = chunks_path
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model_name
        self.embedding_model: SentenceTransformer | None = None
        self.index = None
        self.chunks: list[FaissChunk] = []
        self.chunk_position_by_id: dict[str, int] = {}
        self.chunk_terms: list[set[str]] = []
        self.term_to_chunk_ids: dict[str, set[int]] = {}
        self.signature = ""

    def load(self) -> "FaissKnowledgeBase":
        if faiss is None:
            raise ImportError(
                "FAISS nÃ£o estÃ¡ instalado. Instale `faiss-cpu` para usar a nova arquitetura SBERT + FAISS."
            )

        self.chunks = load_chunks(self.chunks_path)
        self.chunk_position_by_id = {chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)}
        self._build_lexical_index()
        self.signature = f"{len(self.chunks)}-{self.chunks_path.stat().st_mtime_ns}"
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "meta.json"

        if index_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") == self.signature and meta.get("embedding_model_name") == self.embedding_model_name:
                self.index = faiss.read_index(str(index_path))
                return self

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(
            json.dumps(
                {
                    "signature": self.signature,
                    "embedding_model_name": self.embedding_model_name,
                    "chunks_total": len(self.chunks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self

    def _build_lexical_index(self) -> None:
        self.chunk_terms = []
        self.term_to_chunk_ids = {}
        for idx, chunk in enumerate(self.chunks):
            terms = set(tokenize_for_search(chunk.text))
            self.chunk_terms.append(terms)
            for term in terms:
                self.term_to_chunk_ids.setdefault(term, set()).add(idx)

    def search(self, question: str, top_k: int = 8) -> list[RetrievedPassage]:
        if self.index is None:
            self.load()
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        query_vector = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)
        results: list[RetrievedPassage] = []
        seen_docs: dict[str, int] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[int(idx)]
            if seen_docs.get(chunk.doc_id, 0) >= 2:
                continue
            seen_docs[chunk.doc_id] = seen_docs.get(chunk.doc_id, 0) + 1
            results.append(RetrievedPassage(chunk=chunk, score=float(score)))
        return results

    def hybrid_search(self, question: str, top_k: int = 8) -> list[RetrievedPassage]:
        semantic_results = self.search(question, top_k=max(top_k * 3, 12))
        query_terms = expand_query_terms(question)
        if not query_terms:
            return semantic_results[:top_k]

        semantic_by_idx: dict[int, float] = {}
        for item in semantic_results:
            idx = self.chunk_position_by_id.get(item.chunk.chunk_id)
            if idx is not None:
                semantic_by_idx[idx] = float(item.score)

        lexical_counts: dict[int, float] = {}
        for term in query_terms:
            for idx in self.term_to_chunk_ids.get(term, set()):
                lexical_counts[idx] = lexical_counts.get(idx, 0.0) + 1.0

        if not lexical_counts:
            return semantic_results[:top_k]

        max_semantic = max(semantic_by_idx.values(), default=1.0)
        max_lexical = max(lexical_counts.values(), default=1.0)
        candidate_ids = set(semantic_by_idx) | set(sorted(lexical_counts, key=lexical_counts.get, reverse=True)[:80])

        ranked: list[tuple[float, int]] = []
        for idx in candidate_ids:
            chunk = self.chunks[idx]
            terms = self.chunk_terms[idx] if idx < len(self.chunk_terms) else set()
            semantic_score = semantic_by_idx.get(idx, 0.0) / max_semantic if max_semantic else 0.0
            lexical_score = lexical_counts.get(idx, 0.0) / max_lexical if max_lexical else 0.0
            coverage_score = len(terms.intersection(query_terms)) / max(len(query_terms), 1)
            bonus = 0.15 if coverage_score >= 0.5 else 0.0
            if "registro de software" in normalize_text(question) and "registro" in terms and "software" in terms:
                bonus += 0.2
            if any(term in normalize_text(question) for term in ("medicina", "saude")) and (
                "medicina" in terms or "saude" in terms
            ):
                bonus += 0.15
            score = (0.55 * semantic_score) + (0.30 * lexical_score) + (0.15 * coverage_score) + bonus
            ranked.append((score, idx))

        ranked.sort(reverse=True)
        results: list[RetrievedPassage] = []
        seen_docs: dict[str, int] = {}
        for score, idx in ranked:
            chunk = self.chunks[idx]
            if seen_docs.get(chunk.doc_id, 0) >= 2:
                continue
            seen_docs[chunk.doc_id] = seen_docs.get(chunk.doc_id, 0) + 1
            results.append(RetrievedPassage(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break
        return results


class FAISSRAGChatbot:
    def __init__(
        self,
        knowledge_base: FaissKnowledgeBase,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        question_guide: QuestionGuide | None = None,
        faq_bank: FAQAnswerBank | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.question_guide = question_guide
        self.faq_bank = faq_bank
        self.llm = InstructionLLM(llm_model_name)
        self.use_llm = os.getenv("RAG_FAISS_USE_LLM", "1" if DEFAULT_USE_LLM else "0") == "1"

    def answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        retrieve_k: int = 8,
    ) -> dict[str, object]:
        history = history or []
        guide_matches = self.question_guide.match(question, top_k=3) if self.question_guide else []
        follow_up_suggestions = self._build_follow_up_suggestions(question, guide_matches)
        faq_match, faq_score = self.faq_bank.match(question) if self.faq_bank else (None, 0.0)
        if faq_match is not None and faq_score >= 0.78:
            answer = self._format_faq_answer(faq_match, follow_up_suggestions)
            sources = self._faq_sources(faq_match)
            suggested_category = faq_match.category
            similar_questions = [
                {"question": faq_match.question, "category": faq_match.category, "score": round(faq_score, 4)}
            ]
            return {
                "answer": answer,
                "sources": sources,
                "suggested_category": suggested_category,
                "similar_questions": similar_questions,
                "follow_up_suggestions": follow_up_suggestions,
            }
        hinted_question = self.question_guide.build_retrieval_hint(question) if self.question_guide else question
        retrieved = self.knowledge_base.hybrid_search(hinted_question, top_k=retrieve_k)
        if self.use_llm:
            raw_answer = self.llm.generate(self._build_prompt(question, history, retrieved, guide_matches, faq_match))
            answer = self._append_follow_up(self._postprocess_answer(raw_answer), follow_up_suggestions)
        else:
            answer = self._append_follow_up(self._fallback_answer(question, retrieved), follow_up_suggestions)
        return {
            "answer": answer,
            "sources": self._build_sources(retrieved),
            "suggested_category": guide_matches[0].category if guide_matches and guide_matches[0].score >= 0.18 else "",
            "similar_questions": [
                {"question": item.question, "category": item.category, "score": round(item.score, 4)}
                for item in guide_matches
                if item.score >= 0.12
            ],
            "follow_up_suggestions": follow_up_suggestions,
        }

    def _build_prompt(
        self,
        question: str,
        history: list[dict[str, str]],
        retrieved: list[RetrievedPassage],
        guide_matches: list[QuestionMatch],
        faq_match: FAQEntry | None = None,
    ) -> str:
        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {normalize_spaces(turn.get('content', ''))}"
            for turn in history[-4:]
            if normalize_spaces(turn.get("content", ""))
        )
        category_hint = ""
        if faq_match is not None and faq_match.category:
            category_hint = faq_match.category
        elif guide_matches and guide_matches[0].score >= 0.18:
            category_hint = guide_matches[0].category
        context = "\n\n".join(
            [
                f"[Trecho {idx}]\n"
                f"Documento: {item.chunk.doc_id}\n"
                f"Origem: {item.chunk.source_path}\n"
                f"Rotulo: {item.chunk.label}\n"
                f"Texto: {item.chunk.text}"
                for idx, item in enumerate(retrieved, start=1)
            ]
        )
        return (
            "Voce e um assistente RAG especializado na AGIPI, UEPG, AGEUNI e normas do INPI.\n"
            "Responda somente com base nos trechos recuperados.\n"
            "Nao misture a resposta com exemplos, perguntas de treino, dialogos em ingles ou texto fora do dominio AGIPI/UEPG/INPI.\n"
            "Se a informacao nao estiver claramente nos trechos, diga isso explicitamente.\n"
            "Se a pergunta pedir para verificar se existe algum registro, documento, software, patente ou parceria, diga claramente se encontrou evidencia direta ou se nao encontrou nos trechos recuperados.\n"
            "Responda de forma natural, como em uma conversa, sem mencionar documentos, trechos, contexto recuperado, base, busca, FAISS, embeddings, anexos ou processo interno.\n"
            "Entregue 1 ou 2 paragrafos claros e um fechamento curto convidando a pessoa a continuar a conversa.\n"
            "No fechamento, sugira de 2 a 3 formas parecidas de perguntar sobre o mesmo tema.\n\n"
            f"Categoria sugerida: {category_hint or 'nao identificada'}\n\n"
            f"Historico:\n{history_text or 'Sem historico.'}\n\n"
            f"Pergunta:\n{question}\n\n"
            f"Trechos recuperados:\n{context}\n\n"
            "Resposta final:"
        )

    def _format_faq_answer(self, entry: FAQEntry, follow_up_suggestions: list[str]) -> str:
        answer = clean_user_facing_answer(normalize_spaces(entry.answer))
        answer = self._append_follow_up(answer, follow_up_suggestions)
        return answer

    def _faq_sources(self, entry: FAQEntry) -> list[dict[str, object]]:
        normalized = entry.source_docs.replace(";", ",")
        source_docs = [normalize_spaces(part) for part in normalized.split(",") if normalize_spaces(part)]
        if not source_docs:
            source_docs = ["faq_agipi_ageuni_documentos"]
        return [
            {
                "doc_id": doc_id,
                "label": entry.category or "FAQ curada",
                "source_path": "faq_agipi_ageuni_documentos.xlsx",
                "score": 1.0,
                "excerpt": entry.evidence or entry.answer,
            }
            for doc_id in source_docs
        ]

    def _postprocess_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        stop_markers = [
            "\nHuman:",
            "\nUser:",
            "\nAssistant:",
            "\nPerguntas semelhantes:",
            "\nQuestion:",
            "\nAnswer:",
            "How can I improve my English speaking skills?",
        ]
        for marker in stop_markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return "Nao encontrei trechos suficientes na base para responder com seguranca."
        compact = "\n".join(lines)
        compact = compact.removeprefix("Resposta:").strip()
        compact = compact.removeprefix("Resposta final:").strip()
        compact = compact.removeprefix("Resposta Final:").strip()
        paragraphs = [normalize_spaces(part) for part in compact.split("\n") if normalize_spaces(part)]
        if not paragraphs:
            return "Nao encontrei trechos suficientes na base para responder com seguranca."
        if len(paragraphs) == 1:
            return clean_user_facing_answer(paragraphs[0])
        return "\n\n".join(clean_user_facing_answer(part) for part in paragraphs[:3])

    def _fallback_answer(self, question: str, retrieved: list[RetrievedPassage]) -> str:
        if not retrieved:
            return "Nao encontrei trechos suficientes na base para responder com seguranca."
        normalized_question = normalize_text(question)
        query_terms = set(expand_query_terms(question))
        if query_terms:
            best_overlap = max(
                (
                    len(self.knowledge_base.chunk_terms[self.knowledge_base.chunk_position_by_id[item.chunk.chunk_id]].intersection(query_terms))
                    for item in retrieved
                    if item.chunk.chunk_id in self.knowledge_base.chunk_position_by_id
                ),
                default=0,
            )
            existence_intent = any(token in normalized_question for token in ("tem algum", "existe", "ha algum"))
            if existence_intent:
                required_groups: list[set[str]] = []
                if any(token in normalized_question for token in ("software", "programa de computador", "registro de software")):
                    required_groups.append({"software", "programa", "computador", "registro"})
                if any(token in normalized_question for token in ("medicina", "saude", "medico", "medica", "clinica", "hospital")):
                    required_groups.append({"medicina", "saude", "medico", "medica", "clinica", "hospital"})

                if required_groups:
                    has_direct_evidence = False
                    for item in retrieved:
                        idx = self.knowledge_base.chunk_position_by_id.get(item.chunk.chunk_id)
                        if idx is None:
                            continue
                        terms = self.knowledge_base.chunk_terms[idx]
                        if all(any(term in terms for term in group) for group in required_groups):
                            has_direct_evidence = True
                            break
                    if not has_direct_evidence:
                        if len(required_groups) >= 2:
                            return clean_user_facing_answer(
                                (
                                "Nao encontrei uma evidencia direta que ligue todos os elementos da sua pergunta, "
                                "como registro de software e medicina na UEPG."
                                )
                            )
                        return clean_user_facing_answer("Nao encontrei uma evidencia direta para confirmar isso com seguranca.")
                elif best_overlap < 2:
                    return clean_user_facing_answer("Nao encontrei uma evidencia direta para confirmar isso com seguranca.")
        retrieved_text = " ".join(item.chunk.text[:500] for item in retrieved[:3])
        retrieved_normalized = normalize_text(retrieved_text)

        if "registro de software" in normalized_question or (
            "software" in normalized_question and "registro" in normalized_question
        ):
            if "uepg" in normalized_question:
                return clean_user_facing_answer(
                    "Encontrei indicios de que a UEPG trabalha com pedidos de registro de software e com rotinas de avaliacao preliminar para esse tipo de protecao."
                )
            return clean_user_facing_answer(
                "Encontrei indicios relevantes sobre registro de software e sobre os procedimentos relacionados a esse tipo de protecao."
            )

        if "software" in normalized_question or "programa de computador" in normalized_question:
            return clean_user_facing_answer(
                "Encontrei informacoes relacionadas a software e a protecao de programas de computador dentro do contexto de propriedade intelectual e inovacao."
            )

        if "patente" in normalized_question:
            return clean_user_facing_answer(
                "Encontrei informacoes relevantes sobre patenteamento, protecao da inovacao e etapas relacionadas ao pedido de patente."
            )

        if any(term in normalized_question for term in ("agipi", "ageuni", "epitec", "inprotec")):
            return clean_user_facing_answer(
                "Encontrei informacoes consistentes sobre esse tema e posso detalhar melhor o papel institucional, os objetivos ou os servicos envolvidos."
            )

        if "licenciamento" in retrieved_normalized or "transferencia de tecnologia" in retrieved_normalized:
            return clean_user_facing_answer(
                "Encontrei informacoes sobre transferencia de tecnologia, licenciamento e relacionamento entre universidade e parceiros externos."
            )

        return clean_user_facing_answer(
            "Encontrei informacoes relevantes sobre esse assunto e posso aprofundar em um aspecto mais especifico, se voce quiser."
        )

    def _build_follow_up_suggestions(self, question: str, guide_matches: list[QuestionMatch], limit: int = 3) -> list[str]:
        if not self.question_guide:
            return []
        target_category = ""
        if guide_matches and guide_matches[0].score >= 0.18:
            target_category = guide_matches[0].category

        suggestions: list[str] = []
        normalized_question = normalize_question_key(question)
        for example in self.question_guide.examples:
            if target_category and example.category != target_category:
                continue
            if normalize_question_key(example.question) == normalized_question:
                continue
            if example.question not in suggestions:
                suggestions.append(example.question)
            if len(suggestions) >= limit:
                break
        return suggestions

    def _append_follow_up(self, answer: str, follow_up_suggestions: list[str]) -> str:
        cleaned = answer.strip()
        cleaned = clean_user_facing_answer(cleaned)
        if not follow_up_suggestions:
            return cleaned
        if "Se quiser" in cleaned or "Voce tambem pode perguntar" in cleaned:
            return cleaned
        suggestions_text = "; ".join(follow_up_suggestions[:3])
        return (
            f"{cleaned}\n\n"
            f"Se quiser, eu tambem posso continuar por caminhos parecidos, como: {suggestions_text}."
        )

    def _build_sources(self, retrieved: list[RetrievedPassage]) -> list[dict[str, object]]:
        sources: list[dict[str, object]] = []
        seen_docs = set()
        for item in retrieved:
            if item.chunk.doc_id in seen_docs:
                continue
            seen_docs.add(item.chunk.doc_id)
            sources.append(
                {
                    "doc_id": item.chunk.doc_id,
                    "label": item.chunk.label,
                    "source_path": item.chunk.source_path,
                    "score": round(item.score, 4),
                    "excerpt": item.chunk.text[:500],
                }
            )
        return sources


def build_faiss_chatbot_from_env(base_path: Path) -> FAISSRAGChatbot:
    prepared_dir = ensure_prepared_corpus(base_path)
    embedding_model_name = os.getenv("RAG_FAISS_SBERT_MODEL", DEFAULT_SBERT_MODEL)
    llm_model_name = os.getenv("RAG_FAISS_LLM_MODEL", DEFAULT_LLM_MODEL)

    kb = FaissKnowledgeBase(
        chunks_path=prepared_dir / "chunks.jsonl",
        index_dir=base_path / "saida" / "faiss_rag_index",
        embedding_model_name=embedding_model_name,
    ).load()

    question_guide = None
    questions_path = base_path / "Perguntas.xlsx"
    if questions_path.exists():
        try:
            question_guide = QuestionGuide.from_excel(questions_path)
        except Exception:
            question_guide = None

    faq_bank = None
    faq_path = base_path / "faq_agipi_ageuni_documentos.xlsx"
    if faq_path.exists():
        try:
            faq_bank = FAQAnswerBank.from_excel(faq_path)
        except Exception:
            faq_bank = None

    return FAISSRAGChatbot(
        knowledge_base=kb,
        llm_model_name=llm_model_name,
        question_guide=question_guide,
        faq_bank=faq_bank,
    )
