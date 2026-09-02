import json
import os
import re

INPUT_FILE = "data/processed/bis_text/bis_documents.json"
OUTPUT_DIR = "data/processed/bis_chunks"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bis_chunks.json")


def clean_text(text):
    """Clean PDF extraction noise while preserving useful content."""

    if not text:
        return ""

    # Normalize whitespace
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_standard_metadata(text):
    """
    Extract BIS standard number and title from the beginning
    of the extracted document text.
    """

    standard_number = ""
    title = ""

    # ---------------------------------------------------------
    # Extract standard number
    # ---------------------------------------------------------

    number_match = re.match(
        r"(IS\s+\d+(?::\s*\d{4})?)\s+",
        text,
        re.IGNORECASE
    )

    if not number_match:
        return standard_number, title

    standard_number = number_match.group(1).strip()

    # ---------------------------------------------------------
    # Remove standard number
    # ---------------------------------------------------------

    remaining_text = text[number_match.end():].strip()

    # ---------------------------------------------------------
    # Find description start
    # ---------------------------------------------------------

    description_markers = [
        " Wet land cultivation,",
        " The ",
        " This standard ",
        " This Indian Standard ",
        " This document ",
        " This specification ",
        " This code "
    ]

    positions = []

    for marker in description_markers:

        position = remaining_text.lower().find(
            marker.lower()
        )

        if position > 0:
            positions.append(position)

    # ---------------------------------------------------------
    # Extract title
    # ---------------------------------------------------------

    if positions:

        title_end = min(positions)

        title = remaining_text[:title_end].strip()

    else:

        # Fallback: first sentence
        sentence_match = re.search(
            r"\.",
            remaining_text
        )

        if sentence_match:
            title = remaining_text[
                :sentence_match.start()
            ].strip()

    # ---------------------------------------------------------
    # Clean title formatting
    # ---------------------------------------------------------

    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    # Remove accidental description text after the title
    revision_match = re.search(
        r"(.+?\(First Revision\))",
        title,
        re.IGNORECASE
    )

    if revision_match:
        title = revision_match.group(1).strip()

    return standard_number, title


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

        # Extract metadata from document text
        standard_number, title = extract_standard_metadata(text)

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
                "standard_number": standard_number,
                "title": title,
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