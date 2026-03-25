from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from document_utils import (
    chunk_words,
    clean_document_text,
    find_documents_dir,
    infer_label_from_filename,
    infer_label_from_text,
    load_json_documents,
    read_text_file_robust,
    slugify_text,
)


def build_documents(base_path: Path) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []

    json_path = base_path / "documentos_texto.json"
    if json_path.exists():
        for position, (original_id, text) in enumerate(load_json_documents(json_path).items(), start=1):
            if not text:
                continue
            label = infer_label_from_text(text, fallback=original_id)
            documents.append(
                {
                    "doc_id": f"json_{position:04d}",
                    "original_doc_id": original_id,
                    "corpus": "Base Geral JSON",
                    "source_type": "json",
                    "label": label,
                    "source_path": str(json_path),
                    "text": text,
                }
            )

    inpi_dir = find_documents_dir(base_path)
    if inpi_dir:
        for position, file_path in enumerate(sorted(inpi_dir.rglob("*.txt")), start=1):
            raw_text = read_text_file_robust(file_path)
            text = clean_document_text(raw_text)
            if not text:
                continue
            label = infer_label_from_filename(file_path)
            documents.append(
                {
                    "doc_id": f"inpi_{position:04d}",
                    "original_doc_id": file_path.stem,
                    "corpus": "Atos Normativos INPI",
                    "source_type": "txt",
                    "label": label,
                    "source_path": str(file_path),
                    "text": text,
                }
            )

    for doc in documents:
        text = str(doc["text"])
        doc["word_count"] = len(text.split())
        doc["char_count"] = len(text)
        doc["sha1"] = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        doc["slug"] = slugify_text(f"{doc['corpus']}-{doc['label']}-{doc['doc_id']}")

    return documents


def build_chunks(documents: list[dict[str, object]], chunk_size: int, overlap: int) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for doc in documents:
        pieces = chunk_words(str(doc["text"]), chunk_size=chunk_size, overlap=overlap)
        for idx, piece in enumerate(pieces, start=1):
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{idx:04d}",
                    "doc_id": doc["doc_id"],
                    "original_doc_id": doc["original_doc_id"],
                    "corpus": doc["corpus"],
                    "label": doc["label"],
                    "source_type": doc["source_type"],
                    "source_path": doc["source_path"],
                    "chunk_index": idx,
                    "start_word": piece["start_word"],
                    "end_word": piece["end_word"],
                    "text": piece["text"],
                }
            )
    return chunks


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_manifest(documents: list[dict[str, object]], chunks: list[dict[str, object]], chunk_size: int, overlap: int) -> dict[str, object]:
    docs_by_corpus: dict[str, int] = {}
    chunks_by_corpus: dict[str, int] = {}
    for doc in documents:
        docs_by_corpus[str(doc["corpus"])] = docs_by_corpus.get(str(doc["corpus"]), 0) + 1
    for chunk in chunks:
        chunks_by_corpus[str(chunk["corpus"])] = chunks_by_corpus.get(str(chunk["corpus"]), 0) + 1

    return {
        "documents_total": len(documents),
        "chunks_total": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "corpora": [
            {
                "corpus": corpus,
                "documents": docs_by_corpus[corpus],
                "chunks": chunks_by_corpus.get(corpus, 0),
            }
            for corpus in sorted(docs_by_corpus)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara uma base documental unificada para uso futuro em chatbot, RAG ou fine-tuning."
    )
    parser.add_argument("--base-path", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("saida/prepared_corpus"))
    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args()

    base_path = args.base_path.resolve()
    output_dir = (base_path / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()

    documents = build_documents(base_path)
    if not documents:
        raise FileNotFoundError("Nenhum documento foi encontrado para preparar a base.")

    chunks = build_chunks(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    manifest = build_manifest(documents, chunks, chunk_size=args.chunk_size, overlap=args.overlap)

    write_jsonl(output_dir / "documents.jsonl", documents)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Documentos preparados: {manifest['documents_total']}")
    print(f"Chunks gerados: {manifest['chunks_total']}")
    print(f"Saída: {output_dir}")
    for corpus_info in manifest["corpora"]:
        print(
            f"- {corpus_info['corpus']}: "
            f"{corpus_info['documents']} documentos, {corpus_info['chunks']} chunks"
        )


if __name__ == "__main__":
    main()
