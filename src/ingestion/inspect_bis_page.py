from playwright.sync_api import sync_playwright
import json
import os

URL = "https://standards.bis.gov.in/website/published-standards/department-wise"

requests_found = []

with sync_playwright() as p:

    browser = p.chromium.launch(headless=True)

    page = browser.new_page()

    # --------------------------------------------------
    # CAPTURE API RESPONSES
    # --------------------------------------------------

    def capture_response(response):

        request = response.request
        url = response.url

        if request.method == "POST" and (
            "standardsadmin.bis.gov.in" in url
            or "standardsmodule.bis.gov.in" in url
        ):

            try:
                body = response.text()
            except Exception:
                body = ""

            record = {
                "method": request.method,
                "url": url,
                "status": response.status,
                "resource_type": request.resource_type,
                "post_data": request.post_data,
                "response_body": body
            }

            requests_found.append(record)

            print("\n" + "=" * 80)
            print("API:", url)
            print("STATUS:", response.status)

            if request.post_data:
                print("\nPOST DATA:")
                print(request.post_data[:5000])

            print("\nRESPONSE:")
            print(body[:8000])

    page.on("response", capture_response)

    # --------------------------------------------------
    # OPEN BIS
    # --------------------------------------------------

    print("Opening BIS...")

    page.goto(
        URL,
        wait_until="domcontentloaded",
        timeout=120000
    )

    print("TITLE:", page.title())
    print("URL:", page.url)

    page.wait_for_timeout(10000)

    # --------------------------------------------------
    # FIND ETD
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("SEARCHING FOR ETD")
    print("=" * 80)

    etd_expand = page.get_by_role(
        "button",
        name="Expand committees for ELECTROTECHNICAL DEPARTMENT (ETD)"
    )

    print("ETD EXPAND COUNT:", etd_expand.count())

    if etd_expand.count() == 0:

        print("Could not find ETD expand button.")

        browser.close()
        raise SystemExit

    # --------------------------------------------------
    # EXPAND ETD
    # --------------------------------------------------

    print("Expanding ETD...")

    etd_expand.click()

    page.wait_for_timeout(5000)

    # --------------------------------------------------
    # FIND ETD 32
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("SEARCHING FOR ETD 32")
    print("=" * 80)

    committee_text = page.get_by_text(
        "ETD 32 - Electrical Appliances",
        exact=True
    )

    print(
        "ETD 32 COUNT:",
        committee_text.count()
    )

    if committee_text.count() == 0:

        print("ETD 32 not found.")

        browser.close()
        raise SystemExit

    print("ETD 32 found.")

    # --------------------------------------------------
    # INSPECT PARENT HTML
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("ETD 32 HTML")
    print("=" * 80)

    element = committee_text.first

    print(
        element.evaluate(
            "(el) => el.parentElement.outerHTML"
        )[:10000]
    )

    # --------------------------------------------------
    # FIND LINKS INSIDE ETD 32 ROW
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("LINKS AROUND ETD 32")
    print("=" * 80)

    parent = element.locator("xpath=..")

    links = parent.locator("a")

    print("LINK COUNT:", links.count())

    for i in range(links.count()):

        link = links.nth(i)

        print(
            f"\nLINK {i}"
        )

        print(
            "TEXT:",
            link.inner_text()
        )

        print(
            "HREF:",
            link.get_attribute("href")
        )

        print(
            "ARIA:",
            link.get_attribute("aria-label")
        )

    # --------------------------------------------------
    # ALSO CHECK NEARBY BUTTONS
    # --------------------------------------------------

    buttons = parent.locator("button")

    print("\nBUTTON COUNT:", buttons.count())

    for i in range(buttons.count()):

        button = buttons.nth(i)

        print(
            f"\nBUTTON {i}"
        )

        print(
            "TEXT:",
            button.inner_text()
        )

        print(
            "ARIA:",
            button.get_attribute("aria-label")
        )

    # --------------------------------------------------
    # SAVE HTML
    # --------------------------------------------------

    os.makedirs("data/raw", exist_ok=True)

    with open(
        "data/raw/bis_etd32_inspection.html",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(page.content())

    print("\nSaved:")
    print("data/raw/bis_etd32_inspection.html")

    # --------------------------------------------------
    # SCREENSHOT
    # --------------------------------------------------

    page.screenshot(
        path="data/raw/bis_etd32_inspection.png",
        full_page=True
    )

    print("Saved:")
    print("data/raw/bis_etd32_inspection.png")

    # --------------------------------------------------
    # KEEP OPEN A LITTLE LONGER
    # --------------------------------------------------

    page.wait_for_timeout(3000)

    browser.close()


# ------------------------------------------------------
# SAVE API RESPONSES
# ------------------------------------------------------

api_output = "data/raw/bis_etd32_api_responses.json"

with open(
    api_output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        requests_found,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n" + "=" * 80)
print("TOTAL API REQUESTS:", len(requests_found))
print("=" * 80)

print("Saved:", api_output)