from playwright.sync_api import sync_playwright
import json
import os
from datetime import datetime

URL = "https://standards.bis.gov.in/website/published-standards/department-wise"

requests_found = []

with sync_playwright() as p:

    browser = p.chromium.launch(headless=True)

    page = browser.new_page()

    # ---------------------------------------------------------
    # CAPTURE ALL NETWORK REQUESTS
    # ---------------------------------------------------------

    def capture_response(response):

        request = response.request
        url = response.url

        # Ignore obvious static files
        ignored = (
            ".js",
            ".css",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".woff",
            ".woff2",
            ".ttf",
            ".ico",
            ".gif",
        )

        if any(url.lower().split("?")[0].endswith(x) for x in ignored):
            return

        # Capture API-looking requests
        if (
            "standardsadmin.bis.gov.in" in url
            or "standardsmodule.bis.gov.in" in url
        ):

            try:
                body = response.text()
            except Exception:
                body = ""

            data = {
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "url": url,
                "status": response.status,
                "resource_type": request.resource_type,
                "post_data": request.post_data,
                "response_body": body,
            }

            requests_found.append(data)

            print("\n" + "=" * 100)
            print("API REQUEST")
            print("=" * 100)

            print("METHOD :", request.method)
            print("URL    :", url)
            print("STATUS :", response.status)
            print("TYPE   :", request.resource_type)

            if request.post_data:
                print("\nPOST DATA:")
                print(request.post_data[:5000])

            print("\nRESPONSE:")
            print(body[:8000])

    page.on("response", capture_response)

    # ---------------------------------------------------------
    # OPEN PAGE
    # ---------------------------------------------------------

    print("\nOpening BIS Published Standards...")

    page.goto(
        URL,
        wait_until="domcontentloaded",
        timeout=120000
    )

    print("Page loaded.")
    print("TITLE:", page.title())
    print("URL:", page.url)

    # Allow Angular to finish
    page.wait_for_timeout(10000)

    # ---------------------------------------------------------
    # PRINT PAGE TEXT
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("PAGE TEXT")
    print("=" * 100)

    text = page.locator("body").inner_text()

    print(text[:15000])

    # ---------------------------------------------------------
    # PRINT BUTTONS
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("BUTTONS")
    print("=" * 100)

    buttons = page.locator("button")

    for i in range(buttons.count()):
        try:
            print(i, "|", repr(buttons.nth(i).inner_text()))
        except:
            pass

    # ---------------------------------------------------------
    # PRINT SELECT ELEMENTS
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("SELECT ELEMENTS")
    print("=" * 100)

    selects = page.locator("select")

    for i in range(selects.count()):

        try:
            print("\nSELECT", i)

            options = selects.nth(i).locator("option")

            for j in range(options.count()):
                print(
                    j,
                    "|",
                    repr(options.nth(j).inner_text()),
                    "| value=",
                    options.nth(j).get_attribute("value")
                )

        except Exception as e:
            print("ERROR:", e)

    # ---------------------------------------------------------
    # FIND ELECTROTECHNICAL
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("LOOKING FOR ELECTROTECHNICAL")
    print("=" * 100)

    locator = page.get_by_text(
        "Electrotechnical",
        exact=True
    )

    print("COUNT:", locator.count())

    if locator.count() > 0:

        for i in range(locator.count()):

            element = locator.nth(i)

            print("\nELEMENT", i)

            try:
                print("TAG:", element.evaluate("(e) => e.tagName"))
            except:
                pass

            try:
                print("TEXT:", repr(element.inner_text()))
            except:
                pass

            try:
                print(
                    "HTML:",
                    element.evaluate("(e) => e.outerHTML")[:3000]
                )
            except:
                pass

    # ---------------------------------------------------------
    # CLICK ELECTROTECHNICAL
    # ---------------------------------------------------------

    if locator.count() > 0:

        print("\nClicking Electrotechnical...")

        locator.first.scroll_into_view_if_needed()

        locator.first.click(
            force=True
        )

        print("Clicked.")

        page.wait_for_timeout(5000)

    # ---------------------------------------------------------
    # PRINT CURRENT URL
    # ---------------------------------------------------------

    print("\nCurrent URL:", page.url)

    # ---------------------------------------------------------
    # TAKE SCREENSHOT
    # ---------------------------------------------------------

    os.makedirs("data/raw", exist_ok=True)

    page.screenshot(
        path="data/raw/bis_after_click.png",
        full_page=True
    )

    # ---------------------------------------------------------
    # CLOSE
    # ---------------------------------------------------------

    browser.close()


# -------------------------------------------------------------
# SAVE CAPTURED REQUESTS
# -------------------------------------------------------------

os.makedirs("data/raw", exist_ok=True)

output = "data/raw/bis_api_responses.json"

with open(
    output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        requests_found,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n" + "=" * 100)
print("TOTAL API REQUESTS:", len(requests_found))
print("=" * 100)

print("\nSaved:", output)

print("\nScreenshot:")
print("data/raw/bis_after_click.png")