from pathlib import Path

from src.retrieval.embedding.embed_chunks import (
    embed_texts,
    extract_texts_and_metadata,
    load_jsonl,
    save_embeddings,
    save_metadata,
)


INPUT_JSONL = Path("data/intermediate/chunks/papers/papers_basic_sample.jsonl")
OUTPUT_EMBEDDINGS = Path("data/intermediate/embeddings/papers_basic_sample_embeddings.npy")
OUTPUT_METADATA = Path("data/intermediate/embeddings/papers_basic_sample_metadata.json")


def main() -> None:
    print(f"Loading chunks from: {INPUT_JSONL}")
    records = load_jsonl(INPUT_JSONL)
    print(f"Loaded records: {len(records)}")

    texts, metadata = extract_texts_and_metadata(records)
    print(f"Texts to embed: {len(texts)}")

    embeddings = embed_texts(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    save_embeddings(embeddings, OUTPUT_EMBEDDINGS)
    save_metadata(metadata, OUTPUT_METADATA)

    print(f"Saved embeddings to: {OUTPUT_EMBEDDINGS}")
    print(f"Saved metadata to: {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
