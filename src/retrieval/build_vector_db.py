import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/processed/bis_chunks/bis_chunks.json"
OUTPUT_DIR = "data/vector_db"

INDEX_FILE = os.path.join(OUTPUT_DIR, "bis.index")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")


def main():

    print("Loading BIS chunks...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Chunks:", len(chunks))

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    # ---------------------------------------------------------
    # Prepare text
    # ---------------------------------------------------------

    texts = []

    for chunk in chunks:

        text = (
            chunk.get("title", "")
            + "\n"
            + chunk.get("text", "")
        )

        texts.append(text)

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------

    print()
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    print("Embedding shape:", embeddings.shape)

    # ---------------------------------------------------------
    # Normalize embeddings
    # ---------------------------------------------------------

    faiss.normalize_L2(embeddings)

    # ---------------------------------------------------------
    # Create FAISS index
    # ---------------------------------------------------------

    dimension = embeddings.shape[1]

    print()
    print("Creating FAISS index...")
    print("Vector dimension:", dimension)

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("Vectors added:", index.ntotal)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    faiss.write_index(
        index,
        INDEX_FILE
    )

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 60)
    print("VECTOR DATABASE CREATED")
    print("=" * 60)

    print("Vectors:", index.ntotal)
    print("Dimension:", dimension)
    print("Index:", INDEX_FILE)
    print("Metadata:", METADATA_FILE)


if __name__ == "__main__":
    main()