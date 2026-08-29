import pymupdf
import glob
import os
import json

PDF_DIR = "data/raw/bis/pdfs"
OUTPUT_DIR = "data/processed/bis_text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

print("=" * 60)
print("BIS PDF TEXT EXTRACTION")
print("=" * 60)

print(f"PDFs found: {len(pdf_files)}")
print()

records = []

for pdf_path in pdf_files:

    filename = os.path.basename(pdf_path)

    print(f"Processing: {filename}")

    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text
        })

    full_text = "\n\n".join(
        page["text"] for page in pages
    )

    record = {
        "source_file": filename,
        "pages": len(doc),
        "characters": len(full_text),
        "text": full_text,
        "page_text": pages
    }

    records.append(record)

    output_file = os.path.join(
        OUTPUT_DIR,
        filename.replace(".pdf", ".json")
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            record,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"  Pages: {len(doc)} | "
        f"Characters: {len(full_text)}"
    )

    doc.close()


# Save combined dataset
combined_file = os.path.join(
    OUTPUT_DIR,
    "bis_documents.json"
)

with open(
    combined_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print("=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(f"Documents: {len(records)}")
print(f"Saved to: {OUTPUT_DIR}")
print(f"Combined dataset: {combined_file}")