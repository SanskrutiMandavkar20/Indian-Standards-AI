from playwright.sync_api import sync_playwright
import json
import os

URL = "https://standards.bis.gov.in/website/published-standards"

requests_found = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    page = browser.new_page()

    def capture_response(response):
        request = response.request
        url = response.url

        # Capture API/backend requests
        if "/api/" in url.lower():
            requests_found.append({
                "method": request.method,
                "url": url,
                "status": response.status,
                "resource_type": request.resource_type,
                "post_data": request.post_data
            })

    page.on("response", capture_response)

    print("Opening BIS Published Standards...")
    
    page.goto(
        URL,
        wait_until="domcontentloaded",
        timeout=120000
    )

    print("Initial page loaded.")

    # Give Angular time to load lazy modules/API data
    page.wait_for_timeout(15000)

    print("Finished waiting.")

    browser.close()


os.makedirs("data/raw", exist_ok=True)

output = "data/raw/bis_api_requests.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(requests_found, f, indent=2, ensure_ascii=False)


print()
print("=" * 70)
print("API REQUESTS FOUND:", len(requests_found))
print("=" * 70)

for item in requests_found:
    print()
    print(item["method"], "|", item["status"])
    print(item["url"])

    if item["post_data"]:
        print("POST DATA:")
        print(item["post_data"][:1000])

print()
print("Saved:", output)