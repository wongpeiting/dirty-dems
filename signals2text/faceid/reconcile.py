"""Reconcile the independent vision-model cast against the tool's cast, per video.

Reads  vlm_cast_<acct>.json  (the second-coder's per-video names) and
       presence_<acct>.csv    (the tool's per-video cast),
writes  adjudication_<acct>.json  with, per video:
  agree     : both methods named this roster person  → accepted true appearance
  tool_only : tool named them, vision model didn't    → human confirms (precision)
  vlm_only  : vision model named them, tool didn't     → human confirms (a MISS = recall)
  offroster : vision-model names not on the roster     → roster-gap candidates (§1)

Name matching is by slug, with a small VERIFIED alias map for nickname/formal
variants the roster stores differently (Chuck→Charles Schumer, Jim→James Clyburn,
Doug→Douglas Emhoff, …). Aliases are only added after confirming the target slug
is actually on the roster — an unverified alias silently breaks real matches.

  python reconcile.py whitehouse
  python reconcile.py democrats
"""
import json
import sys
from pathlib import Path

import pandas as pd

import config
from faceid import io_utils

# CANON: fold variant/duplicate slugs onto ONE canonical slug. Applied to BOTH the
# tool's cast and the vision model's names, so a person split across two slugs (a
# roster duplicate) or spelled two ways (a nickname) is counted once on each side.
#   - roster DUPLICATES (same person, two roster rows): doug/douglas Emhoff, ben/benjamin Cardin
#   - VLM nickname → roster canonical: Chuck→Charles Schumer, James→Jim Clyburn, AOC
# Verify each target is a real roster slug before adding.
CANON = {
    "doug-emhoff": "douglas-emhoff",
    "benjamin-l-cardin": "ben-cardin",
    "chuck-schumer": "charles-schumer",
    "james-clyburn": "jim-clyburn",
    "aoc": "alexandria-ocasio-cortez",
}


def roster_slugs():
    return {io_utils.slugify(n): n for n in pd.read_csv(config.WATCHLIST_DIR / "roster.csv")["name"]}


def reconcile(account, verbose=True):
    roster = roster_slugs()
    canon = {k: v for k, v in CANON.items() if v in roster}
    vlm = json.loads((config.OUTPUT_DIR / f"vlm_cast_{account}.json").read_text())
    pres = pd.read_csv(config.OUTPUT_DIR / f"presence_{account}.csv", dtype={"video_id": str})

    def c(slug):
        return canon.get(slug, slug)

    tool_by_vid = {v: {c(p) for p in g["person"]} for v, g in pres.groupby("video_id")}

    rows, offall = [], {}
    for vid, names in vlm.items():
        vs, unknown, off = set(), False, []
        for n in names:
            if n == "unknown-face":
                unknown = True
                continue
            s = c(io_utils.slugify(n))
            (vs.add(s) if s in roster else off.append(n))
        for o in off:
            offall[o] = offall.get(o, 0) + 1
        tool = tool_by_vid.get(vid, set())
        rows.append(dict(video_id=vid, agree=sorted(vs & tool),
                         vlm_only=sorted(vs - tool), tool_only=sorted(tool - vs),
                         unknown_face=unknown, offroster=off))
    (config.OUTPUT_DIR / f"adjudication_{account}.json").write_text(json.dumps(rows, indent=1))

    if verbose:
        nag = sum(len(r["agree"]) for r in rows)
        nvo = sum(len(r["vlm_only"]) for r in rows)
        nto = sum(len(r["tool_only"]) for r in rows)
        print(f"=== reconcile @{account} ({len(rows)} videos) ===")
        print(f"  AGREEMENTS: {nag}   VLM-only (candidate MISS): {nvo}   TOOL-only (confirm): {nto}")
        print(f"  → {nvo + nto} adjudication cards; off-roster names: "
              f"{sorted(offall.items(), key=lambda x: -x[1])}")
    return rows


if __name__ == "__main__":
    reconcile(sys.argv[1] if len(sys.argv) > 1 else "whitehouse")
