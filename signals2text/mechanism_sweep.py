"""
Mechanism-detection sweep (Abidin-derived), read-only & local.
Detects candidate attention mechanisms per post from Stage-1 descriptions + OCR +
captions + music fingerprints, then prints per-account frequency tables so the ~5%
rule can decide which earn a slot in the Mechanisms dimension. Saves per-post hits to
data/eval/mechanism_sweep.json (for the review GUI / later charts).

Sources per mechanism (Abidin, TikTok & Youth Cultures, 2025):
  screenshot_attack   Ch3 p.71 green screens as receipts; F36 scroll-attacks
  no_context          Ch2 p.62-64 'No context'/IYKYK; F38 emoji attacks
  call_to_engage      Ch2 p.49-51 'Please Interact'
  cliffhanger         Ch2 p.60-61 'Part X'/'follow for part two'
  tags_opponent       Ch2 p.64-66 Voldemorting (inverse: does the attack NAME the target account?)
  duet_stitch         Ch3 p.70 duet chains; stitch replies
  lip_sync            Ch3 p.69 memetic practice #1
  filter_distortion   Ch3 p.73-75 filters as mockery; deep-fried edits
  second_person       Ch2 p.56-57 second-person address solicits interaction
  static_card         Ch3 p.78 cross-platform capital (screenshot-ready text cards)
  trending_sound      F41 participate-vs-broadcast (fingerprint context)
"""
import json, glob, csv, re, os, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
ACCTS = ["whitehouse", "democrats", "republicans"]
DESC = os.path.join(ROOT, "data/descriptions/gemini_v2")
OCR = os.path.join(ROOT, "data/ocr")
MUSIC = os.path.join(ROOT, "data/music")

EMOJI_ONLY = re.compile(r"^[^\w#@]*$", re.UNICODE)  # caption w/ no word chars at all
WORD = re.compile(r"[A-Za-z0-9]")

def caps():
    d = {}
    for a in ACCTS:
        # posts CSV first (full corpus incl. pre-window + new posts; caption is in 'title')
        for r in csv.DictReader(open(os.path.join(ROOT, f"data/{a}_posts.csv"))):
            d[(a, str(r["id"]))] = r.get("title") or r.get("caption") or ""
        # coding sheet overrides where present (curated caption column)
        p = os.path.join(ROOT, f"{a}_coding_sheet.csv")
        for r in csv.DictReader(open(p)):
            if (r.get("caption") or "").strip():
                d[(a, r["video_id"])] = r["caption"]
    return d

def blob(d, *keys):
    out = []
    for k in keys:
        v = d.get(k)
        if isinstance(v, str): out.append(v)
        elif isinstance(v, list):
            for x in v:
                out.append(x if isinstance(x, str) else json.dumps(x))
    return " ".join(out).lower()

def detect(a, d, cap, ocr_text, fp):
    hits = []
    vis = blob(d, "visual_action", "editing_pacing", "spoken_summary")
    claim = blob(d, "central_claim", "intended_effect")
    manip = (d.get("manipulation_or_satire") or "").lower()
    ost = " ".join(t.get("text", "") for t in (d.get("on_screen_text") or [])).lower()
    capl = cap.lower()
    everything = " ".join([vis, claim, ost, capl, ocr_text])

    # A. engagement bait
    if re.search(r"screenshot|screen record|scroll(ing|s)? through|green.?screen|truth social post|tweet(ed)? (from|by)|points? (at|to) (a|the|an)? ?(post|tweet|headline|article|screenshot)", vis + " " + claim):
        hits.append("screenshot_attack")
    spoken_empty = not WORD.search(d.get("spoken_summary") or "") or bool(re.search(r"no (spoken|speech|dialogue)|no narration|instrumental only", (d.get("spoken_summary") or "").lower()))
    if cap.strip() and EMOJI_ONLY.match(cap.replace("#", "").replace("@", "").strip()):
        hits.append("no_context")          # emoji-only caption
    elif len(cap.strip()) < 12 and spoken_empty and not ost.strip():
        hits.append("no_context")          # near-captionless, silent, no overlay
    if re.search(r"comment (if|below|your)|tag (a|someone|your)|share (this|if)|like (if|this video)|drop (a|your)|let us know|tell us|duet this|stitch this|sound off|reply with", capl + " " + ost):
        hits.append("call_to_engage")
    if re.search(r"part (two|2|\d+)|wait for it|to be continued|follow for (more|part)|stay tuned", capl + " " + ost):
        hits.append("cliffhanger")
    if re.search(r"\byou(r|'re)?\b", capl) or re.search(r"\byou(r|'re)?\b", ost[:200]):
        hits.append("second_person")

    # B. attack delivery
    opp = {"democrats": r"@(whitehouse|realdonaldtrump|gop|republicans|jdvance|elonmusk)",
           "whitehouse": r"@(democrats|thedemocrats|joebiden|kamalaharris|aoc|hakeemjeffries|chuckschumer|govpritzker|gavinnewsom)",
           "republicans": r"@(democrats|thedemocrats|joebiden|kamalaharris|aoc|hakeemjeffries|chuckschumer)"}[a]
    if re.search(opp, capl):
        hits.append("tags_opponent")
    if re.search(r"#?\b(duet|stitch)\b|split.?screen|side.by.side (with|video)", capl + " " + vis):
        hits.append("duet_stitch")

    # C. participatory craft
    if re.search(r"lip.?sync|mouth(s|ing) (along|the words)|mim(es|ing) (the|to)", vis):
        hits.append("lip_sync")
    if re.search(r"deep.?fried|distort(ed|ion)|warp(ed)?|face filter|beauty filter|exaggerat\w+ (features|face)|cartoonish filter", manip + " " + vis):
        hits.append("filter_distortion")
    if re.search(r"static (image|graphic|text)|single still|text card|no video footage|slideshow of text|graphic card", vis):
        hits.append("static_card")
    if fp:
        hits.append("trending_sound")      # fingerprint found a real track
    return hits

def main():
    captions = caps()
    per = {a: collections.Counter() for a in ACCTS}
    n = {a: 0 for a in ACCTS}
    examples = collections.defaultdict(list)
    rows = []
    for a in ACCTS:
        for f in sorted(glob.glob(os.path.join(DESC, a, "*.json"))):
            d = json.load(open(f)); vid = d.get("video_id") or os.path.basename(f)[:-5]
            n[a] += 1
            ocr_text = ""
            op = os.path.join(OCR, a, f"{vid}.json")
            if os.path.exists(op):
                try: ocr_text = (json.load(open(op)).get("full_text") or "").lower()
                except Exception: pass
            fp = []
            mp = os.path.join(MUSIC, a, f"{vid}.json")
            if os.path.exists(mp):
                try: fp = json.load(open(mp)).get("distinct_tracks") or []
                except Exception: pass
            hits = detect(a, d, captions.get((a, vid), ""), ocr_text, fp)
            for h in hits:
                per[a][h] += 1
                if len(examples[h]) < 6: examples[h].append(f"{a}/{vid}")
            rows.append({"account": a, "video_id": vid, "mechanisms": hits})

    MECHS = ["screenshot_attack","no_context","call_to_engage","cliffhanger","second_person",
             "tags_opponent","duet_stitch","lip_sync","filter_distortion","static_card","trending_sound"]
    print(f"{'mechanism':18} {'WH':>10} {'DEM':>10} {'GOP':>10}   verdict vs 5% bar")
    print("-" * 72)
    for m in MECHS:
        cells = []
        for a in ACCTS:
            pct = 100 * per[a][m] / max(1, n[a])
            cells.append(f"{per[a][m]:4} {pct:4.1f}%")
        mx = max(100 * per[a][m] / max(1, n[a]) for a in ACCTS)
        verdict = "PASS" if mx >= 5 else "cut"
        print(f"{m:18} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}   {verdict}")
    out = os.path.join(ROOT, "data/eval/mechanism_sweep.json")
    json.dump({"counts": {a: dict(per[a]) for a in ACCTS}, "n": n,
               "examples": dict(examples), "posts": rows}, open(out, "w"), indent=0)
    print(f"\nsaved per-post hits → {out}")

if __name__ == "__main__":
    main()
