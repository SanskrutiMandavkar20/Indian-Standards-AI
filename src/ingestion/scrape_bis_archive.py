import csv
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


URL = "https://www.bis.gov.in/archive-standard-of-the-month/?lang=en"

OUTPUT = "data/raw/bis/standards_metadata.csv"


def extract_standard_number(title):
    """
    Extract IS / IS/ISO / SP number from the title.
    """
    patterns = [
        r"\bIS\s*/?\s*ISO\s*[\w()/:.-]+",
        r"\bIS\s*[\w()/:.-]+",
        r"\bSP\s*[\w()/:.-]+",
    ]

    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return ""


def scrape():
    print("Downloading BIS archive...")

    response = requests.get(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=60
    )

    print("Status:", response.status_code)
    print("Size:", len(response.text))

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table")

    if not table:
        print("ERROR: Could not find standards table.")
        return

    rows = table.find_all("tr")

    print("Rows found:", len(rows))

    records = []

    for row in rows:
        cells = row.find_all(["td", "th"])

        if len(cells) < 6:
            continue

        values = [
            cell.get_text(" ", strip=True)
            for cell in cells
        ]

        # Skip header
        if values[0].lower() in ["sr.no", "sr. no.", "sr no"]:
            continue

        title = values[1]

        links = row.find_all("a")

        view_url = ""
        download_url = ""

        for link in links:
            href = link.get("href")

            if not href:
                continue

            href = urljoin(URL, href)

            text = link.get_text(" ", strip=True).lower()

            if "download" in text:
                download_url = href

            elif "view" in text:
                view_url = href

        record = {
            "sr_no": values[0],
            "standard_number": extract_standard_number(title),
            "title": title,
            "size": values[2],
            "format": values[3],
            "language": values[4],
            "publish_date": values[5],
            "view_url": view_url,
            "download_url": download_url,
            "source": URL,
        }

        records.append(record)

    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    print()
    print("=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print("Records:", len(records))
    print("Saved:", OUTPUT)

    print()
    print("First 5 records:")

    for record in records[:5]:
        print(
            record["standard_number"],
            "|",
            record["title"]
        )


if __name__ == "__main__":
    scrape()