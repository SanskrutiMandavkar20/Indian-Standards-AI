from pathlib import Path
import json
import re


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_ROOT = BASE_DIR / "data" / "processed" / "cleaned"
OUTPUT_ROOT = BASE_DIR / "data" / "processed" / "chunks"

STANDARD_INPUT_DIR = INPUT_ROOT / "standards"
CERTIFICATION_INPUT_DIR = INPUT_ROOT / "certification"

STANDARD_OUTPUT_DIR = OUTPUT_ROOT / "standards"
CERTIFICATION_OUTPUT_DIR = OUTPUT_ROOT / "certification"

STANDARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CERTIFICATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Approximate target chunk size in characters.
# We deliberately keep chunks fairly large so specification
# requirements are not separated unnecessarily.
TARGET_CHARS = 1800

# Small overlap helps preserve context between chunks.
OVERLAP_CHARS = 250


# ============================================================
# REGEX
# ============================================================

STANDARD_NUMBER_PATTERN = re.compile(
    r"\bIS\s+\d{4,6}(?:\s*\([^)]{1,100}\))?\s*:\s*\d{4}\b",
    re.IGNORECASE
)


# ============================================================
# HELPERS
# ============================================================

def normalize_standard_number(value):
    """
    Normalize small formatting differences in IS numbers.
    Example:
        IS 17873 : 2022
        IS 17873:2022
    ->  IS 17873:2022
    """

    if not value:
        return None

    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"\s*:\s*", ":", value)

    return value


def detect_standard_numbers(text):
    """
    Find all Indian Standard numbers mentioned in a text block.
    """
    matches = STANDARD_NUMBER_PATTERN.findall(text or "")

    normalized = []

    for match in matches:
        value = normalize_standard_number(match)

        if value and value not in normalized:
            normalized.append(value)

    return normalized


def clean_chunk_text(text):
    """
    Final lightweight normalization for chunks.
    Do NOT aggressively rewrite the source.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    # Normalize unusual spaces
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")

    # Normalize horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_long_text(text, target_chars=TARGET_CHARS):
    """
    Split very large page text into overlapping character windows.

    We prefer paragraph boundaries where possible.
    """

    text = clean_chunk_text(text)

    if len(text) <= target_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current) + len(paragraph) + 2 <= target_chars:
            if current:
                current += "\n\n" + paragraph
            else:
                current = paragraph
        else:
            if current:
                chunks.append(current.strip())

            # If one paragraph itself is too large,
            # split it using overlapping windows.
            if len(paragraph) > target_chars:

                start = 0

                while start < len(paragraph):

                    end = min(
                        start + target_chars,
                        len(paragraph)
                    )

                    piece = paragraph[start:end].strip()

                    if piece:
                        chunks.append(piece)

                    if end >= len(paragraph):
                        break

                    start = max(
                        end - OVERLAP_CHARS,
                        start + 1
                    )

                current = ""

            else:
                current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def extract_document_title(document):
    """
    Try to obtain a useful title from the first page.

    This is intentionally conservative.
    We do not try to invent titles.
    """

    pages = document.get("pages", [])

    if not pages:
        return None

    first_page_text = pages[0].get("text", "").strip()

    if not first_page_text:
        return None

    lines = [
        line.strip()
        for line in first_page_text.splitlines()
        if line.strip()
    ]

    # Look for a line containing an IS number.
    for line in lines[:40]:

        if STANDARD_NUMBER_PATTERN.search(line):
            return line[:500]

    # Otherwise use the first reasonably sized line.
    for line in lines[:20]:

        if 10 <= len(line) <= 250:
            return line

    return None


# ============================================================
# STANDARD DOCUMENT CHUNKING
# ============================================================

def chunk_standard_document(document):
    """
    Chunk an Indian Standard while preserving page-level evidence.
    """

    source_pdf = document.get("source_pdf")
    document_type = document.get("document_type", "standard")

    document_title = extract_document_title(document)

    chunks = []

    chunk_counter = 0

    pages = document.get("pages", [])

    for page in pages:

        page_number = page.get("page_number")
        page_text = page.get("text", "")

        page_text = clean_chunk_text(page_text)

        if not page_text:
            continue

        page_standard_numbers = detect_standard_numbers(page_text)

        page_chunks = split_long_text(page_text)

        for local_index, chunk_text in enumerate(page_chunks):

            if not chunk_text:
                continue

            chunk_counter += 1

            chunk_standard_numbers = detect_standard_numbers(
                chunk_text
            )

            # If the chunk itself doesn't contain the number,
            # inherit the number detected on the page.
            if not chunk_standard_numbers:
                chunk_standard_numbers = page_standard_numbers.copy()

            chunk = {
                "chunk_id": (
                    f"{Path(source_pdf).stem}"
                    f"_p{page_number}"
                    f"_c{local_index + 1}"
                ),

                "document_type": document_type,

                "source_pdf": source_pdf,

                "document_title": document_title,

                "page_number": page_number,

                "chunk_index": chunk_counter,

                "text": chunk_text,

                "character_count": len(chunk_text),

                "standard_numbers": chunk_standard_numbers,

                "evidence": {
                    "source_pdf": source_pdf,
                    "page_number": page_number
                }
            }

            chunks.append(chunk)

    return chunks


# ============================================================
# CERTIFICATION DOCUMENT CHUNKING
# ============================================================

def chunk_certification_document(document):
    """
    Chunk BIS certification/regulatory documents.

    These are kept logically separate from product standards.
    """

    source_pdf = document.get("source_pdf")
    document_type = document.get("document_type", "certification")

    document_title = extract_document_title(document)

    chunks = []

    chunk_counter = 0

    pages = document.get("pages", [])

    for page in pages:

        page_number = page.get("page_number")
        page_text = clean_chunk_text(page.get("text", ""))

        if not page_text:
            continue

        page_standard_numbers = detect_standard_numbers(page_text)

        page_chunks = split_long_text(page_text)

        for local_index, chunk_text in enumerate(page_chunks):

            if not chunk_text:
                continue

            chunk_counter += 1

            chunk_standard_numbers = detect_standard_numbers(
                chunk_text
            )

            if not chunk_standard_numbers:
                chunk_standard_numbers = page_standard_numbers.copy()

            chunk = {
                "chunk_id": (
                    f"{Path(source_pdf).stem}"
                    f"_p{page_number}"
                    f"_c{local_index + 1}"
                ),

                "document_type": document_type,

                "source_pdf": source_pdf,

                "document_title": document_title,

                "page_number": page_number,

                "chunk_index": chunk_counter,

                "text": chunk_text,

                "character_count": len(chunk_text),

                "standard_numbers": chunk_standard_numbers,

                "evidence": {
                    "source_pdf": source_pdf,
                    "page_number": page_number
                }
            }

            chunks.append(chunk)

    return chunks


# ============================================================
# PROCESS DIRECTORY
# ============================================================

def process_directory(input_dir, output_dir, chunk_function):

    json_files = sorted(input_dir.glob("*.json"))

    # Ignore reports
    json_files = [
        path for path in json_files
        if path.name != "extraction_report.json"
    ]

    print(f"\nInput: {input_dir}")
    print(f"Documents found: {len(json_files)}")

    total_chunks = 0
    successful_documents = 0

    for json_path in json_files:

        print(f"\nProcessing: {json_path.name}")

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:
                document = json.load(f)

            if not document.get("success", False):
                print("  SKIPPED: extraction was unsuccessful")
                continue

            chunks = chunk_function(document)

            output_data = {
                "success": True,
                "source_pdf": document.get("source_pdf"),
                "document_type": document.get("document_type"),
                "document_title": extract_document_title(document),
                "chunk_count": len(chunks),
                "chunks": chunks
            }

            output_path = output_dir / json_path.name

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    output_data,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(f"  Chunks created: {len(chunks)}")

            total_chunks += len(chunks)
            successful_documents += 1

        except Exception as e:

            print(
                f"  ERROR: {type(e).__name__}: {e}"
            )

    print("\n----------------------------------------")
    print(f"Successful documents: {successful_documents}")
    print(f"Total chunks: {total_chunks}")
    print("----------------------------------------")

    return total_chunks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STAGE 3 — DOCUMENT CHUNKING")
    print("=" * 60)

    print("\nProcessing Indian Standards...")
    standard_chunks = process_directory(
        STANDARD_INPUT_DIR,
        STANDARD_OUTPUT_DIR,
        chunk_standard_document
    )

    print("\nProcessing Certification Documents...")
    certification_chunks = process_directory(
        CERTIFICATION_INPUT_DIR,
        CERTIFICATION_OUTPUT_DIR,
        chunk_certification_document
    )

    # --------------------------------------------------------
    # Combined report
    # --------------------------------------------------------

    report = {
        "standards_chunks": standard_chunks,
        "certification_chunks": certification_chunks,
        "total_chunks": (
            standard_chunks +
            certification_chunks
        )
    }

    report_path = OUTPUT_ROOT / "chunking_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(
        f"\nStandards chunks:      {standard_chunks}"
    )

    print(
        f"Certification chunks: {certification_chunks}"
    )

    print(
        f"Total chunks:          "
        f"{standard_chunks + certification_chunks}"
    )

    print(
        f"\nReport saved to:\n{report_path}"
    )


if __name__ == "__main__":
    main()