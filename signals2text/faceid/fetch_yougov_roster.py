"""Pull the FULL YouGov fame-ranked Republican + Democrat rosters (name + portrait)
from the public-data API, download each portrait as a face reference, build embeddings.

YouGov gives both the name and an entity portrait per person — used here only as an
internal face-recognition reference (not republished). Orgs/no-face entries auto-drop
(InsightFace finds no face → refs_embed skips them).
"""
import csv
import time

import requests

import config
from faceid import io_utils, refs_embed

BASE = "https://api-test.yougov.com/public-data/v5/us/search/entity/"
GROUPS = {"R": "073bb3b6-adf0-11e9-8bb2-373b0b3b3eb4",
          "D": "07016957-adf0-11e9-9161-317b338eee4b"}
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_group(party, group):
    out = []
    offset = 0
    while offset < 400:
        r = requests.get(BASE, params={"group": group, "sort_by": "fame",
                                       "limit": 20, "offset": offset}, headers=H, timeout=30)
        if r.status_code != 200:
            break
        data = r.json().get("data", [])
        if not data:
            break
        for e in data:
            if e.get("name") and e.get("image"):
                out.append({"name": e["name"], "image": e["image"],
                            "party": party, "fame": e.get("fame") or e.get("score")})
        offset += 20
        time.sleep(0.4)
    return out


def main():
    config.WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    roster, seen = [], set()
    for party, g in GROUPS.items():
        ents = fetch_group(party, g)
        print(f"[yougov] {party}: {len(ents)} entities")
        roster += ents
    manifest, got = [], 0
    for e in roster:
        slug = io_utils.slugify(e["name"])
        if slug in seen:
            continue
        seen.add(slug)
        pdir = config.WATCHLIST_DIR / slug
        n = len(list(pdir.glob("*.jpg"))) if pdir.exists() else 0
        if n == 0:
            try:
                img = requests.get(e["image"], headers=H, timeout=25)
                img.raise_for_status()
                pdir.mkdir(parents=True, exist_ok=True)
                (pdir / "01.jpg").write_bytes(img.content)
                with (pdir / "sources.csv").open("w", newline="") as fh:
                    csv.writer(fh).writerow(["file", "url", "licence"])
                    csv.writer(fh).writerow(["01.jpg", e["image"], "YouGov entity image (internal reference)"])
                n = 1
                time.sleep(0.2)
            except Exception as ex:
                print(f"[yougov] {e['name']}: image fail ({ex})")
        manifest.append({"name": e["name"], "slug": slug, "party": e["party"],
                         "fame": e.get("fame"), "photos": n})
        got += 1 if n else 0
    with (config.WATCHLIST_DIR / "roster.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "slug", "party", "fame", "photos"])
        w.writeheader()
        w.writerows(manifest)
    print(f"[yougov] roster: {len(manifest)} people, {got} with a portrait")
    n_emb = refs_embed.build()
    print(f"[yougov] built {n_emb} face references (orgs/no-face auto-dropped) → watchlist_embeddings.npz")


if __name__ == "__main__":
    main()
