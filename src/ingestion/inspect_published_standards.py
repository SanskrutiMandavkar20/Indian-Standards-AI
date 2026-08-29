import requests
import re


JS_URL = (
    "https://standards.bis.gov.in/website/"
    "published-standards/main.67c3adb497a3c85f.js"
)


def main():

    print("Downloading BIS JavaScript...")

    response = requests.get(
        JS_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/120.0 Safari/537.36"
            )
        },
        timeout=60
    )

    response.raise_for_status()

    js = response.text

    print("JavaScript size:", len(js))

    # --------------------------------------------------
    # 1. Find HTTP-looking strings
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("HTTP / URL REFERENCES")
    print("=" * 60)

    patterns = [
        r'https?://[^"\']+',
        r'http[^"\']+',
        r'//[^"\']+',
    ]

    found = set()

    for pattern in patterns:

        matches = re.findall(
            pattern,
            js,
            flags=re.IGNORECASE
        )

        for match in matches:
            found.add(match)

    for item in sorted(found):

        if len(item) < 300:
            print(item)


    # --------------------------------------------------
    # 2. Search useful keywords
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("IMPORTANT KEYWORDS")
    print("=" * 60)

    keywords = [
        "baseUrl",
        "baseURL",
        "apiUrl",
        "apiURL",
        "environment",
        "serviceUrl",
        "serviceURL",
        "endpoint",
        "published",
        "department",
        "search",
        "get(",
        "post(",
        "HttpClient"
    ]

    for keyword in keywords:

        positions = []

        start = 0

        while True:

            position = js.find(keyword, start)

            if position == -1:
                break

            positions.append(position)

            start = position + len(keyword)

        print(
            f"\n{keyword}: {len(positions)} occurrences"
        )

        # Print a few surrounding snippets
        for position in positions[:3]:

            beginning = max(0, position - 150)
            ending = min(
                len(js),
                position + 300
            )

            snippet = js[beginning:ending]

            print("\n", snippet)


if __name__ == "__main__":
    main()