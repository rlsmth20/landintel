from pathlib import Path
from playwright.sync_api import sync_playwright
import time

DOWNLOAD_DIR = Path(r"C:\Users\Rainer\landrisk\data\fema_downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Put the FEMA results-table URL here if you have it bookmarked.
# If not, the script will pause so you can navigate to it manually.
START_URL = "https://hazards.fema.gov/femaportal/NFHL/searchResult"


def safe_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        print("Opening FEMA...")
        page.goto(START_URL, wait_until="domcontentloaded")
        time.sleep(3)

        print("\nIf needed, manually navigate to the FEMA results table page you showed me.")
        print("When you can see the table with county rows and zip icons, press ENTER here.")
        input()

        # Filter to Mississippi using the table search box
        search_box = page.locator('input[type="search"]').first
        search_box.fill("MISSISSIPPI")
        time.sleep(3)

        page_num = 1
        downloaded = set()

        while True:
            print(f"\nProcessing page {page_num}...")

            # Wait for table rows
            page.locator("table tbody tr").first.wait_for(timeout=15000)
            rows = page.locator("table tbody tr")
            row_count = rows.count()
            print(f"Found {row_count} visible rows")

            for i in range(row_count):
                row = rows.nth(i)

                # Read county + item id to build a stable filename
                cells = row.locator("td")
                if cells.count() < 6:
                    continue

                item_id = cells.nth(0).inner_text().strip()
                county = cells.nth(1).inner_text().strip()
                state = cells.nth(2).inner_text().strip()

                key = f"{item_id}_{county}_{state}"
                if key in downloaded:
                    continue

                # The download icon is usually a link in the last column
                download_link = cells.nth(5).locator("a").first

                try:
                    print(f"Downloading: {key}")

                    with page.expect_download(timeout=60000) as dl_info:
                        download_link.click()

                    download = dl_info.value
                    suggested = download.suggested_filename
                    if not suggested.lower().endswith(".zip"):
                        suggested = f"{safe_filename(key)}.zip"

                    save_path = DOWNLOAD_DIR / suggested
                    download.save_as(save_path)

                    print(f"Saved: {save_path}")
                    downloaded.add(key)
                    time.sleep(1)

                except Exception as e:
                    print(f"Failed download for {key}: {e}")

            # Try next page
            next_button = page.locator("a:has-text('Next')")
            if next_button.count() == 0:
                print("No Next button found. Done.")
                break

            # Check if disabled
            parent_html = next_button.first.evaluate("(el) => el.outerHTML")
            if "disabled" in parent_html.lower():
                print("Next button disabled. Done.")
                break

            try:
                next_button.first.click()
                time.sleep(3)
                page_num += 1
            except Exception as e:
                print(f"Could not go to next page: {e}")
                break

        print("\nFinished.")
        browser.close()


if __name__ == "__main__":
    main()