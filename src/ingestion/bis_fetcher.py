import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import json


BIS_URL = "https://www.bis.gov.in/know-your-standard/"


DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_bis_page(url=BIS_URL):
    """
    Fetch the official BIS Know Your Standard page.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return response.text


def inspect_bis_page(html):
    """
    Inspect the BIS page and return basic information.
    """

    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(strip=True) if soup.title else None

    links = []

    for link in soup.find_all("a", href=True):
        text = link.get_text(" ", strip=True)

        if text:
            links.append({
                "text": text,
                "url": link["href"]
            })

    return {
        "page_title": title,
        "link_count": len(links),
        "links": links
    }


def save_raw_data(data, filename="bis_page_inspection.json"):
    """
    Save the inspection result locally.
    """

    output_path = DATA_DIR / filename

    payload = {
        "source": BIS_URL,
        "retrieved_at": datetime.now().isoformat(),
        "data": data
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    print("Connecting to BIS...")

    html = fetch_bis_page()

    print("BIS page downloaded successfully.")

    inspection = inspect_bis_page(html)

    print(
        "Page title:",
        inspection["page_title"]
    )

    print(
        "Links found:",
        inspection["link_count"]
    )

    save_raw_data(inspection)

    print("BIS inspection complete.")