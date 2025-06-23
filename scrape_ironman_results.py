"""
scrape_ironman_results.py
-------------------------
Collect the full results table from an Ironman race page
and save it to ironman_results.csv
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import pandas as pd
import time, os, pathlib

def format_duration_value(val):
    if isinstance(val, str):
        stripped_val = val.strip() # Remove leading/trailing whitespace
        if ':' not in stripped_val: # Early exit if no colon, not a H:MM:SS or MM:SS candidate
            return val

        parts = stripped_val.split(':')
        
        # Check if all components of the split are purely numeric
        # e.g., "57", "12" for "57:12" or "1", "12", "24" for "1:12:24"
        all_parts_numeric = True
        for p_item in parts:
            if not p_item.strip().isdigit(): # Strip each part before checking if it's a digit
                all_parts_numeric = False
                break
        
        if all_parts_numeric:
            if len(parts) == 2:  # Looks like MM:SS format, e.g., "57:12"
                return f"0:{stripped_val}" # Convert to "0:57:12"
            # If len(parts) == 3 (H:MM:SS like "1:12:24"), it's already in the desired a_format_duration_valuet_val.
            # It will fall through and return the original 'val' via the final return.
            # Other cases (e.g., len(parts) != 2 or 3, or non-numeric parts) also fall through.
            
    return val # Return original value if not a string, or not a format we need to change

# ────────────────────────────────────────────────────────────
# 1. CONFIG
# ────────────────────────────────────────────────────────────
TARGET_URL = "https://www.coachcox.co.uk/imstats/race/2156/results/"  # <-- change me
SCROLL_PAUSE = 0.4          # seconds; tune if connection is slow
HEADLESS      = True        # switch to False to watch the browser work
CSV_OUT       = "ironman_results.csv"

# ────────────────────────────────────────────────────────────
# 2. LAUNCH SELENIUM BROWSER
# ────────────────────────────────────────────────────────────
chrome_opts = Options()
if HEADLESS:
    chrome_opts.add_argument("--headless=new")   # Chrome >= 109
chrome_opts.add_argument("--window-size=1920,1080")
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=chrome_opts)
driver.get(TARGET_URL)

# Wait until the table element exists
wait = WebDriverWait(driver, 15)
wait.until(
    EC.presence_of_element_located(
        (By.CSS_SELECTOR, "table#imacresultstable, table#imraceresultstable")
    )
)

# ────────────────────────────────────────────────────────────
# 3. SCROLL UNTIL ALL ROWS ARE LOADED
#    (DataTables lazy‑loads ~50 rows at a time)
# ────────────────────────────────────────────────────────────
scroll_div = wait.until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "div.dt-scroll-body"))
)

last_height, unchanged = 0, 0
while True:
    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_div)
    time.sleep(SCROLL_PAUSE)

    # check if we have reached the end
    current_height = driver.execute_script("return arguments[0].scrollHeight", scroll_div)
    if current_height == last_height:
        unchanged += 1
        if unchanged > 2:   # two consecutive loops with no height change
            break
    else:
        unchanged = 0
        last_height = current_height

print("Scrolling finished – full table loaded.")

# ────────────────────────────────────────────────────────────
# 4. PARSE TABLE WITH BEAUTIFULSOUP
# ────────────────────────────────────────────────────────────
html   = driver.page_source
driver.quit()

soup   = BeautifulSoup(html, "lxml")
table  = soup.select_one("table#imacresultstable, table#imraceresultstable")

headers = [th.get_text(strip=True) for th in table.select("thead th")]
rows    = []
for tr in table.select("tbody tr"):
    cells = [td.get_text(strip=True) for td in tr.select("td")]
    # skip empty rows (if any)
    if any(cells):
        rows.append(cells)

# Fallback header if the JS markup uses <td> in thead (rare)
if not headers:
    first_row_len = len(rows[0])
    headers = [f"Col{i+1}" for i in range(first_row_len)]

# ────────────────────────────────────────────────────────────
# 5. SAVE TO CSV
# ────────────────────────────────────────────────────────────
df = pd.DataFrame(rows, columns=headers[: len(rows[0])])

df = df.applymap(format_duration_value) # Apply the formatting function to all cells

df.to_csv(CSV_OUT, index=False)
print(f"\nDone! {len(df):,} rows written to {CSV_OUT}")