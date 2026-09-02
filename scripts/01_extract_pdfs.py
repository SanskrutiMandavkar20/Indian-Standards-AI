from pathlib import Path
import json
import re
import sys

try:
    import pymupdf
except ImportError:
    print("ERROR: PyMuPDF is not installed.")
    print("Run:")
    print("pip install pymupdf")
    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

STANDARDS_PDF_DIR = BASE_DIR / "data" / "pdfs"
CERTIFICATION_PDF_DIR = BASE_DIR / "data" / "certification"

STANDARDS_OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" / "extracted" / "standards"
)

CERTIFICATION_OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" / "extracted" / "certification"
)

STANDARDS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CERTIFICATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STANDARD NUMBER DETECTION
# ============================================================

STANDARD_PATTERN = re.compile(
    r"\bIS\s+\d{4,6}"
    r"(?:\s*\([^)]{1,100}\))?"
    r"\s*:\s*\d{4}\b",
    re.IGNORECASE
)


def extract_standard_numbers(text):
    """
    Detect Indian Standard numbers present in extracted text.

    This function only detects numbers.
    It does not modify the original PDF text.
    """

    matches = STANDARD_PATTERN.findall(text)

    results = []
    seen = set()

    for match in matches:

        normalized = re.sub(r"\s+", " ", match.strip())

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
# MINIMAL TEXT CLEANING
# ============================================================

def clean_extracted_text(text):
    """
    Minimal cleanup.

    IMPORTANT:
    We do not aggressively rewrite the PDF text at this stage.
    The original extracted evidence must be preserved.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    lines = text.split("\n")

    cleaned_lines = []

    previous_blank = False

    for line in lines:

        line = line.rstrip()

        if not line.strip():

            if not previous_blank:
                cleaned_lines.append("")

            previous_blank = True

        else:

            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()


# ============================================================
# EXTRACT ONE PDF
# ============================================================

def extract_pdf(pdf_path, document_type, output_dir):

    print()
    print("-" * 60)
    print(f"Processing: {pdf_path.name}")
    print(f"Type      : {document_type}")

    try:
        document = pymupdf.open(pdf_path)

    except Exception as exc:

        print(f"ERROR opening PDF: {exc}")

        return {
            "success": False,
            "document_type": document_type,
            "source_pdf": pdf_path.name,
            "page_count": 0,
            "pages_with_text": 0,
            "total_characters": 0,
            "pages": [],
            "error": str(exc)
        }

    pages = []

    total_characters = 0
    pages_with_text = 0

    for page_index in range(len(document)):

        page_number = page_index + 1

        try:

            page = document.load_page(page_index)

            raw_text = page.get_text("text")

            text = clean_extracted_text(raw_text)

            standard_numbers = extract_standard_numbers(text)

            character_count = len(text)

            if text:
                pages_with_text += 1

            total_characters += character_count

            pages.append({
                "page": page_number,
                "text": text,
                "character_count": character_count,
                "standard_numbers": standard_numbers
            })

        except Exception as exc:

            pages.append({
                "page": page_number,
                "text": "",
                "character_count": 0,
                "standard_numbers": [],
                "error": str(exc)
            })

    document.close()

    # --------------------------------------------------------
    # Detect all standards in this PDF
    # --------------------------------------------------------

    detected_standards = []

    for page in pages:

        for standard in page["standard_numbers"]:

            if standard not in detected_standards:
                detected_standards.append(standard)

    result = {
        "success": True,
        "document_type": document_type,
        "source_pdf": pdf_path.name,
        "page_count": len(pages),
        "pages_with_text": pages_with_text,
        "total_characters": total_characters,
        "standards_detected": detected_standards,
        "pages": pages
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output_path = output_dir / f"{pdf_path.stem}.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Pages            : {len(pages)}")
    print(f"Pages with text   : {pages_with_text}")
    print(f"Characters        : {total_characters}")

    if detected_standards:

        print("Standards detected:")

        for standard in detected_standards:
            print(f"  - {standard}")

    print(f"Saved             : {output_path}")

    return result


# ============================================================
# PROCESS A DIRECTORY
# ============================================================

def process_directory(
    pdf_dir,
    output_dir,
    document_type
):

    print()
    print("=" * 60)
    print(f"{document_type.upper()} PDF PROCESSING")
    print("=" * 60)

    if not pdf_dir.exists():

        print()
        print(f"WARNING: Directory does not exist:")
        print(pdf_dir)

        return []

    pdf_files = sorted(
        pdf_dir.glob("*.pdf")
    )

    print()
    print(f"Directory : {pdf_dir}")
    print(f"PDF files : {len(pdf_files)}")

    results = []

    for pdf_path in pdf_files:

        result = extract_pdf(
            pdf_path,
            document_type,
            output_dir
        )

        results.append(result)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("INDIAN STANDARDS AI")
    print("PDF TEXT EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Standards
    # --------------------------------------------------------

    standard_results = process_directory(
        STANDARDS_PDF_DIR,
        STANDARDS_OUTPUT_DIR,
        "standard"
    )

    # --------------------------------------------------------
    # Certification
    # --------------------------------------------------------

    certification_results = process_directory(
        CERTIFICATION_PDF_DIR,
        CERTIFICATION_OUTPUT_DIR,
        "certification"
    )

    # --------------------------------------------------------
    # Combined report
    # --------------------------------------------------------

    all_results = (
        standard_results +
        certification_results
    )

    successful = sum(
        1
        for result in all_results
        if result["success"]
    )

    failed = sum(
        1
        for result in all_results
        if not result["success"]
    )

    total_pages = sum(
        result["page_count"]
        for result in all_results
        if result["success"]
    )

    total_characters = sum(
        result["total_characters"]
        for result in all_results
        if result["success"]
    )

    report = {
        "standards": {
            "pdf_count": len(standard_results),
            "successful": sum(
                1
                for r in standard_results
                if r["success"]
            ),
            "failed": sum(
                1
                for r in standard_results
                if not r["success"]
            )
        },

        "certification": {
            "pdf_count": len(certification_results),
            "successful": sum(
                1
                for r in certification_results
                if r["success"]
            ),
            "failed": sum(
                1
                for r in certification_results
                if not r["success"]
            )
        },

        "overall": {
            "total_pdfs": len(all_results),
            "successful": successful,
            "failed": failed,
            "total_pages": total_pages,
            "total_characters": total_characters
        },

        "files": all_results
    }

    report_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "extracted"
        / "extraction_report.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print()
    print("Standards PDFs")
    print(f"  Found      : {len(standard_results)}")
    print(f"  Successful : {sum(r['success'] for r in standard_results)}")
    print(f"  Failed     : {sum(not r['success'] for r in standard_results)}")

    print()
    print("Certification PDFs")
    print(f"  Found      : {len(certification_results)}")
    print(f"  Successful : {sum(r['success'] for r in certification_results)}")
    print(f"  Failed     : {sum(not r['success'] for r in certification_results)}")

    print()
    print("Overall")
    print(f"  Total PDFs       : {len(all_results)}")
    print(f"  Successful       : {successful}")
    print(f"  Failed           : {failed}")
    print(f"  Total pages      : {total_pages}")
    print(f"  Total characters : {total_characters}")

    print()
    print(f"Report:")
    print(report_path)

    print()
    print("IMPORTANT:")
    print("Do not build the cleaning/chunking/embedding stages yet.")
    print("First verify the extracted PDF content.")


if __name__ == "__main__":
    main()