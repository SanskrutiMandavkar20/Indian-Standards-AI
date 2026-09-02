from pathlib import Path
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_ROOT = BASE_DIR / "data" / "processed" / "chunks"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"

STANDARD_CHUNKS_DIR = CHUNKS_ROOT / "standards"
CERTIFICATION_CHUNKS_DIR = CHUNKS_ROOT / "certification"

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# SETTINGS
# ============================================================

# Number of chunks encoded in one batch.
BATCH_SIZE = 32


# ============================================================
# HELPERS
# ============================================================

def load_chunks(directory):
    """
    Load all chunk JSON files from a directory.
    """

    all_chunks = []

    json_files = sorted(directory.glob("*.json"))

    print(f"\nReading from: {directory}")
    print(f"Files found: {len(json_files)}")

    for json_path in json_files:

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:
                document = json.load(f)

            if not document.get("success", False):
                continue

            chunks = document.get("chunks", [])

            for chunk in chunks:

                text = chunk.get("text", "").strip()

                if not text:
                    continue

                all_chunks.append(chunk)

        except Exception as e:

            print(
                f"ERROR reading {json_path.name}: "
                f"{type(e).__name__}: {e}"
            )

    return all_chunks


def build_embedding_text(chunk):
    """
    Construct the text used for semantic embedding.

    Metadata is included as context, while the original
    chunk text remains untouched in the metadata.
    """

    parts = []

    document_type = chunk.get("document_type")
    document_title = chunk.get("document_title")
    standard_numbers = chunk.get("standard_numbers", [])
    text = chunk.get("text", "")

    if document_type:
        parts.append(f"Document type: {document_type}")

    if document_title:
        parts.append(f"Document title: {document_title}")

    if standard_numbers:
        parts.append(
            "Indian Standards: "
            + ", ".join(standard_numbers)
        )

    parts.append("Source text:")
    parts.append(text)

    return "\n".join(parts)


def create_index(
    chunks,
    index_name,
    metadata_name
):
    """
    Create FAISS inner-product index using normalized embeddings.
    """

    if not chunks:
        print(f"\nNo chunks available for {index_name}")
        return

    print("\n" + "=" * 60)
    print(f"CREATING {index_name.upper()} INDEX")
    print("=" * 60)

    print(f"Chunks: {len(chunks)}")

    # --------------------------------------------------------
    # Prepare embedding text
    # --------------------------------------------------------

    embedding_texts = [
        build_embedding_text(chunk)
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        f"\nLoading embedding model:\n"
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        embedding_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------
    #
    # Because embeddings are normalized,
    # inner product = cosine similarity.
    #

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"FAISS vectors: {index.ntotal}"
    )

    print(
        f"Vector dimension: {dimension}"
    )

    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    index_path = VECTOR_DB_DIR / index_name

    faiss.write_index(
        index,
        str(index_path)
    )

    print(
        f"\nFAISS index saved:\n"
        f"{index_path}"
    )

    # --------------------------------------------------------
    # Build metadata
    # --------------------------------------------------------

    metadata = []

    for vector_id, chunk in enumerate(chunks):

        metadata.append({
            "vector_id": vector_id,

            "chunk_id": chunk.get("chunk_id"),

            "document_type": chunk.get(
                "document_type"
            ),

            "source_pdf": chunk.get(
                "source_pdf"
            ),

            "document_title": chunk.get(
                "document_title"
            ),

            "page_number": chunk.get(
                "page_number"
            ),

            "chunk_index": chunk.get(
                "chunk_index"
            ),

            "character_count": chunk.get(
                "character_count"
            ),

            "standard_numbers": chunk.get(
                "standard_numbers",
                []
            ),

            # Original source text.
            "text": chunk.get("text", ""),

            # Evidence location.
            "evidence": chunk.get(
                "evidence",
                {}
            )
        })

    metadata_path = VECTOR_DB_DIR / metadata_name

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Metadata saved:\n"
        f"{metadata_path}"
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    print("\nVerification:")

    print(
        f"  Metadata entries : {len(metadata)}"
    )

    print(
        f"  FAISS vectors    : {index.ntotal}"
    )

    print(
        f"  Dimensions       : {dimension}"
    )

    if len(metadata) == index.ntotal:
        print("  ✓ Vector/metadata counts match")
    else:
        print("  ✗ ERROR: counts do not match")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STAGE 4 — EMBEDDINGS + FAISS")
    print("=" * 60)

    # --------------------------------------------------------
    # Standards
    # --------------------------------------------------------

    standard_chunks = load_chunks(
        STANDARD_CHUNKS_DIR
    )

    print(
        f"\nTotal standard chunks: "
        f"{len(standard_chunks)}"
    )

    # --------------------------------------------------------
    # Certification
    # --------------------------------------------------------

    certification_chunks = load_chunks(
        CERTIFICATION_CHUNKS_DIR
    )

    print(
        f"Total certification chunks: "
        f"{len(certification_chunks)}"
    )

    # --------------------------------------------------------
    # Create separate indexes
    # --------------------------------------------------------

    create_index(
        chunks=standard_chunks,
        index_name="standards.index",
        metadata_name="standards_metadata.json"
    )

    create_index(
        chunks=certification_chunks,
        index_name="certification.index",
        metadata_name="certification_metadata.json"
    )

    print("\n" + "=" * 60)
    print("STAGE 4 COMPLETE")
    print("=" * 60)

    print("\nCreated:")

    print(
        "  data/vector_db/standards.index"
    )

    print(
        "  data/vector_db/standards_metadata.json"
    )

    print(
        "  data/vector_db/certification.index"
    )

    print(
        "  data/vector_db/certification_metadata.json"
    )


if __name__ == "__main__":
    main()