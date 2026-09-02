import json
from pathlib import Path
import fitz
from tqdm import tqdm


PDF_DIR = Path("data/pdfs")
OUTPUT_FILE = Path("data/extracted/documents.json")


def extract_pdf(pdf_path):
    """
    Extract text page-by-page from a PDF.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return pages


def main():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files.")

    documents = []

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):

        try:

            pages = extract_pdf(pdf_path)

            documents.append({
                "source_pdf": pdf_path.name,
                "pages": pages
            })

        except Exception as e:

            print(f"\nERROR processing {pdf_path.name}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nExtraction complete.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()