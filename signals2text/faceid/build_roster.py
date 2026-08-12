"""Build the politician whitelist references from a YouGov-derived roster.

Roster = YouGov fame-ranked top-20 Republicans + top-20 Democrats + codebook extras.
For each name, fetch the lead portrait from Wikipedia's REST summary API → watchlist/<slug>/.
Then build watchlist_embeddings.npz. Names with no usable face are logged (un-matchable).
"""
import csv
import time
from pathlib import Path

import requests

import config
from faceid import io_utils, refs_embed

ROSTER = [
    # Republicans (YouGov fame top-20)
    ("Donald Trump", "R"), ("Arnold Schwarzenegger", "R"), ("Robert F. Kennedy Jr.", "R"),
    ("JD Vance", "R"), ("George W. Bush", "R"), ("Ted Cruz", "R"), ("Mike Pence", "R"),
    ("Sarah Palin", "R"), ("Mitt Romney", "R"), ("Ron DeSantis", "R"), ("Marco Rubio", "R"),
    ("Dick Cheney", "R"), ("Mitch McConnell", "R"), ("Kash Patel", "R"), ("Liz Cheney", "R"),
    ("Lindsey Graham", "R"), ("Mike Huckabee", "R"), ("Jeb Bush", "R"), ("Newt Gingrich", "R"),
    ("Kristi Noem", "R"),
    # Democrats (YouGov fame top-20)
    ("Hillary Clinton", "D"), ("Joe Biden", "D"), ("Bill Clinton", "D"), ("Barack Obama", "D"),
    ("Kamala Harris", "D"), ("Bernie Sanders", "D"), ("Nancy Pelosi", "D"), ("Al Gore", "D"),
    ("Elizabeth Warren", "D"), ("Tim Walz", "D"), ("Gavin Newsom", "D"),
    ("Alexandria Ocasio-Cortez", "D"), ("Andrew Cuomo", "D"), ("Zohran Mamdani", "D"),
    ("Chuck Schumer", "D"), ("Michael Bloomberg", "D"), ("Pete Buttigieg", "D"),
    ("John Kerry", "D"), ("Ilhan Omar", "D"), ("Hakeem Jeffries", "D"),
    # codebook cast not in the YouGov political top-20s
    ("Michelle Obama", "D"), ("Karoline Leavitt", "R"), ("Pete Hegseth", "R"), ("Elon Musk", "X"),
]

WIKI = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikimedia requires a descriptive User-Agent with contact; full-res images trip robot policy.
HEADERS = {"User-Agent": "pol-face-research/1.0 (pw2635@columbia.edu) requests"}


def _get(url, tries=4):
    for i in range(tries):
        r = requests.get(url, headers=HEADERS, timeout=25)
        if r.status_code == 429:
            time.sleep(3 * (i + 1))   # back off on rate-limit
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def fetch_portrait(name, slug):
    pdir = config.WATCHLIST_DIR / slug
    if pdir.exists() and any(pdir.glob("*.jpg")):
        return len(list(pdir.glob("*.jpg")))
    title = name.replace(" ", "_")
    try:
        j = _get(WIKI + title).json()
        # prefer the pre-sized thumbnail (small → not robot-policy-blocked)
        url = (j.get("thumbnail") or j.get("originalimage") or {}).get("source")
        if not url:
            print(f"[roster] {name}: no image on Wikipedia")
            return 0
        time.sleep(1.0)
        img = _get(url)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "01.jpg").write_bytes(img.content)
        with (pdir / "sources.csv").open("w", newline="") as fh:
            csv.writer(fh).writerow(["file", "url", "licence"])
            csv.writer(fh).writerow(["01.jpg", url, "Wikipedia (see file page)"])
        return 1
    except Exception as e:
        print(f"[roster] {name}: fetch failed ({e})")
        return 0


def main():
    config.WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    got = 0
    for name, party in ROSTER:
        slug = io_utils.slugify(name)
        n = fetch_portrait(name, slug)
        manifest.append({"name": name, "slug": slug, "party": party, "photos": n})
        got += 1 if n else 0
        time.sleep(1.5)
    with (config.WATCHLIST_DIR / "roster.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "slug", "party", "photos"])
        w.writeheader()
        w.writerows(manifest)
    print(f"[roster] fetched portraits for {got}/{len(ROSTER)} politicians")
    n_emb = refs_embed.build()
    print(f"[roster] built {n_emb} reference embeddings → watchlist_embeddings.npz")
    missing = [m["name"] for m in manifest if not m["photos"]]
    if missing:
        print(f"[roster] NO photo for: {missing}")


if __name__ == "__main__":
    main()
