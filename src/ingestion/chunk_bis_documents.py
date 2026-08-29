
import json
import os
import re

INPUT_FILE = "data/processed/bis_text/bis_documents.json"
OUTPUT_DIR = "data/processed/bis_chunks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bis_chunks.json")


def clean_text(text):
    """Clean PDF extraction noise while preserving useful content."""

    # Normalize whitespace
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def split_into_chunks(text, chunk_size=1200, overlap=200):
    """
    Split text into overlapping chunks.

    chunk_size = approximate number of characters per chunk
    overlap = characters shared between adjacent chunks
    """

    words = text.split()

    chunks = []
    current = []
    current_length = 0

    for word in words:

        word_length = len(word) + 1

        if current_length + word_length > chunk_size:

            if current:
                chunks.append(" ".join(current))

            # Keep overlap from previous chunk
            overlap_words = []
            overlap_length = 0

            for w in reversed(current):

                if overlap_length + len(w) + 1 > overlap:
                    break

                overlap_words.insert(0, w)
                overlap_length += len(w) + 1

            current = overlap_words
            current_length = overlap_length

        current.append(word)
        current_length += word_length

    if current:
        chunks.append(" ".join(current))

    return chunks


def main():

    print("Loading BIS documents...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print("Documents:", len(documents))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = []

    chunk_id = 0

    for document in documents:

        source_file = document["source_file"]
        text = clean_text(document["text"])

        chunks = split_into_chunks(text)

        print(
            f"{source_file} | "
            f"characters: {len(text)} | "
            f"chunks: {len(chunks)}"
        )

        for i, chunk in enumerate(chunks):

            chunk_id += 1

            all_chunks.append({
                "chunk_id": f"bis_{chunk_id}",
                "standard_number": document.get(
                    "standard_number",
                    ""
                ),
                "title": document.get(
                    "title",
                    ""
                ),
                "source_file": source_file,
                "chunk_index": i,
                "text": chunk
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)
    print("Documents:", len(documents))
    print("Chunks:", len(all_chunks))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()