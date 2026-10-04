"""Download the FEMP O&M Best Practices Guide and keep only the chapters used in this project."""

from pathlib import Path
from urllib.request import Request, urlopen

from pypdf import PdfReader, PdfWriter

# Official source: U.S. Department of Energy, FEMP O&M Best Practices Guide, Release 3.0
SOURCE_URL = "https://www.energy.gov/sites/default/files/2020/04/f74/omguide_complete_w-eo-disclaimer.pdf"

DATA_DIR = Path("data")
FULL_PDF = DATA_DIR / "femp_om_guide.pdf"
SUBSET_PDF = DATA_DIR / "femp_om_guide_ch3-8.pdf"

# Chapters 3-8 = PDF pages 29-124 (counted from 1, both included)
FIRST_PAGE = 29
LAST_PAGE = 124


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if FULL_PDF.exists():
        print("Full guide already downloaded, skipping download.")
    else:
        print("Downloading the full guide...")
        # Some servers reject requests without a browser-like User-Agent
        request = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            FULL_PDF.write_bytes(response.read())

    reader = PdfReader(FULL_PDF)
    writer = PdfWriter()

    # pypdf counts pages from 0, so page N is reader.pages[N - 1]
    for page_number in range(FIRST_PAGE, LAST_PAGE + 1):
        writer.add_page(reader.pages[page_number - 1])

    with open(SUBSET_PDF, "wb") as f:
        writer.write(f)

    print(f"Saved {LAST_PAGE - FIRST_PAGE + 1} pages to {SUBSET_PDF}")


if __name__ == "__main__":
    main()
