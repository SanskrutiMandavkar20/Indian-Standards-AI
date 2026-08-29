import os
import re
import requests
import pandas as pd
from urllib.parse import urlparse

CSV_PATH="data/raw/bis/standards_metadata.csv"
OUTPUT_DIR="data/raw/bis/pdfs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df=pd.read_csv(CSV_PATH)

print("="*60)
print("BIS PDF DOWNLOADER")
print("="*60)

session=requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

success = 0
failed = 0

for index, row in df.iterrows():

    standard_number = str(row["standard_number"]).strip()
    title = str(row["title"]).strip()
    url = str(row["download_url"]).strip()

    if not url.startswith("http"):
        print(f"\nSKIP: {standard_number}")
        print("Invalid URL:", url)
        failed += 1
        continue

    # Clean filename
    safe_name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        standard_number
    )

    safe_name = safe_name.replace(" ", "_")

    filename = f"{safe_name}.pdf"
    output_path = os.path.join(OUTPUT_DIR, filename)

    print(f"\n[{index + 1}/{len(df)}] {standard_number}")
    print("Downloading:", url)

    try:

        response = session.get(
            url,
            timeout=60
        )

        print(
            "Status:",
            response.status_code,
            "| Size:",
            len(response.content),
            "bytes"
        )

        content_type = response.headers.get(
            "content-type",
            ""
        ).lower()

        if response.status_code == 200:

            # Basic protection against accidentally saving HTML
            if "pdf" in content_type or response.content[:4] == b"%PDF":

                with open(output_path, "wb") as f:
                    f.write(response.content)

                print("Saved:", output_path)
                success += 1

            else:

                print(
                    "WARNING: Response does not look like a PDF."
                )
                print(
                    "Content-Type:",
                    content_type
                )

                failed += 1

        else:

            print(
                "FAILED HTTP STATUS:",
                response.status_code
            )

            failed += 1

    except Exception as e:

        print("ERROR:", e)
        failed += 1


print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print("Successful:", success)
print("Failed:", failed)
print("Total:", len(df))