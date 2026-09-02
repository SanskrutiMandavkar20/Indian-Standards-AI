from pathlib import Path
import json
import re
import sys


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EXTRACTED_DIR = (
    BASE_DIR / "data" / "processed" / "extracted"
)

CLEANED_DIR = (
    BASE_DIR / "data" / "processed" / "cleaned"
)

STANDARDS_EXTRACTED_DIR = EXTRACTED_DIR / "standards"
CERTIFICATION_EXTRACTED_DIR = EXTRACTED_DIR / "certification"

STANDARDS_CLEANED_DIR = CLEANED_DIR / "standards"
CERTIFICATION_CLEANED_DIR = CLEANED_DIR / "certification"

STANDARDS_CLEANED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CERTIFICATION_CLEANED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Conservative PDF text cleaning.

    IMPORTANT:
    This function must NOT rewrite the meaning of the source.

    It only:
      - normalizes line endings
      - removes null characters
      - normalizes whitespace
      - removes excessive blank lines
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Normalize unusual spaces
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")

    # Process lines
    lines = text.split("\n")

    cleaned_lines = []

    previous_blank = False

    for line in lines:

        # Remove leading/trailing whitespace
        line = line.strip()

        # Collapse repeated spaces/tabs
        line = re.sub(r"[ \t]+", " ", line)

        if not line:

            if not previous_blank:
                cleaned_lines.append("")

            previous_blank = True

        else:

            cleaned_lines.append(line)
            previous_blank = False

    # Remove blank lines at beginning/end
    while cleaned_lines and not cleaned_lines[0]:
        cleaned_lines.pop(0)

    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


# ============================================================
# STANDARD NUMBER NORMALIZATION
# ============================================================

STANDARD_PATTERN = re.compile(
    r"\bIS\s+\d{4,6}"
    r"(?:\s*\([^)]{1,100}\))?"
    r"\s*:\s*\d{4}\b",
    re.IGNORECASE
)


def detect_standard_numbers(text):

    if not text:
        return []

    matches = STANDARD_PATTERN.findall(text)

    results = []
    seen = set()

    for match in matches:

        normalized = re.sub(
            r"\s+",
            " ",
            match.strip()
        )

        normalized = re.sub(
            r"^is\b",
            "IS",
            normalized,
            flags=re.IGNORECASE
        )

        key = normalized.lower()

        if key not in seen:

            seen.add(key)
            results.append(normalized)

    return results


# ============================================================
# CLEAN ONE DOCUMENT
# ============================================================

def clean_document(input_path, output_path):

    print()
    print("-" * 60)
    print(f"Cleaning: {input_path.name}")

    try:

        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as f:

            document = json.load(f)

    except Exception as exc:

        print(f"ERROR reading file: {exc}")

        return {
            "success": False,
            "file": input_path.name,
            "error": str(exc)
        }

    if not document.get("success", False):

        print("Skipping failed extraction.")

        return {
            "success": False,
            "file": input_path.name,
            "error": document.get(
                "error",
                "Unknown extraction error"
            )
        }

    cleaned_pages = []

    total_characters_before = 0
    total_characters_after = 0

    pages_with_text = 0

    detected_standards = []

    for page_data in document.get("pages", []):

        page_number = page_data.get("page")

        original_text = page_data.get(
            "text",
            ""
        )

        cleaned = clean_text(original_text)

        total_characters_before += len(
            original_text
        )

        total_characters_after += len(
            cleaned
        )

        if cleaned:
            pages_with_text += 1

        page_standards = detect_standard_numbers(
            cleaned
        )

        for standard in page_standards:

            if standard not in detected_standards:
                detected_standards.append(standard)

        cleaned_pages.append({
            "page": page_number,
            "text": cleaned,
            "character_count": len(cleaned),
            "standard_numbers": page_standards
        })

    # --------------------------------------------------------
    # Preserve important document metadata
    # --------------------------------------------------------

    cleaned_document = {

        "success": True,

        "document_type": document.get(
            "document_type"
        ),

        "source_pdf": document.get(
            "source_pdf"
        ),

        "page_count": document.get(
            "page_count",
            len(cleaned_pages)
        ),

        "pages_with_text": pages_with_text,

        "characters_before_cleaning":
            total_characters_before,

        "characters_after_cleaning":
            total_characters_after,

        "standards_detected":
            detected_standards,

        "pages": cleaned_pages
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cleaned_document,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Characters before : "
        f"{total_characters_before}"
    )

    print(
        f"Characters after  : "
        f"{total_characters_after}"
    )

    print(
        f"Pages with text   : "
        f"{pages_with_text}"
    )

    if detected_standards:

        print("Standards detected:")

        for standard in detected_standards:
            print(f"  - {standard}")

    print(f"Saved             : {output_path}")

    return cleaned_document


# ============================================================
# PROCESS DIRECTORY
# ============================================================

def process_directory(
    input_dir,
    output_dir
):

    if not input_dir.exists():

        print()
        print(
            f"WARNING: Directory does not exist:"
        )
        print(input_dir)

        return []

    files = sorted(
        input_dir.glob("*.json")
    )

    # Do not treat extraction_report.json as a document
    files = [
        path
        for path in files
        if path.name != "extraction_report.json"
    ]

    print()
    print(f"Directory: {input_dir}")
    print(f"JSON files: {len(files)}")

    results = []

    for input_path in files:

        output_path = (
            output_dir /
            input_path.name
        )

        result = clean_document(
            input_path,
            output_path
        )

        results.append(result)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("INDIAN STANDARDS AI")
    print("TEXT CLEANING")
    print("=" * 60)

    # --------------------------------------------------------
    # Standards
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("STANDARDS")
    print("=" * 60)

    standard_results = process_directory(
        STANDARDS_EXTRACTED_DIR,
        STANDARDS_CLEANED_DIR
    )

    # --------------------------------------------------------
    # Certification
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CERTIFICATION")
    print("=" * 60)

    certification_results = process_directory(
        CERTIFICATION_EXTRACTED_DIR,
        CERTIFICATION_CLEANED_DIR
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    all_results = (
        standard_results +
        certification_results
    )

    successful = sum(
        1
        for result in all_results
        if result.get("success")
    )

    failed = len(all_results) - successful

    print()
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print()
    print("Standards:")
    print(
        f"  Documents processed: "
        f"{len(standard_results)}"
    )

    print()
    print("Certification:")
    print(
        f"  Documents processed: "
        f"{len(certification_results)}"
    )

    print()
    print("Overall:")
    print(f"  Successful: {successful}")
    print(f"  Failed    : {failed}")

    print()
    print("Output:")

    print(
        f"  Standards     : "
        f"{STANDARDS_CLEANED_DIR}"
    )

    print(
        f"  Certification : "
        f"{CERTIFICATION_CLEANED_DIR}"
    )

    print()
    print("IMPORTANT:")
    print("Do not build the chunking stage yet.")
    print("Verify the cleaned text first.")


if __name__ == "__main__":
    main()