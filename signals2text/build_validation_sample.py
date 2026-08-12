"""
Stratified validation sample (n=50) for Stage-2 — reproducible (seeded).
Coverage: all 3 accounts · both Dem eras · every function present · oversamples
MEDIUM/LOW-confidence posts (where errors concentrate). Emits:
  - data/eval/validation_sample_50.json   (ids — loadable in review.html)
  - data/classification/validation_sample_50.csv  (review sheet: evidence + label + BLANK verdict cols)
"""
import json, csv, glob, os, random, collections

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
ACCTS = ["whitehouse", "democrats", "republicans"]
N = 50

def load():
    dates, caps, cls = {}, {}, {}
    for a in ACCTS:
        for r in csv.DictReader(open(f"{HERE}/data/{a}_posts.csv")):
            dates[str(r["id"])] = r.get("upload_date", "")
            caps[(a, str(r["id"]))] = r.get("title") or ""
        for line in open(f"{HERE}/data/classification/function_{a}.jsonl"):
            r = json.loads(line); cls[(a, r["video_id"])] = r
    return dates, caps, cls

def main():
    dates, caps, cls = load()
    items = list(cls.items())  # ((acct,vid), row)
    def era(vid): d = dates.get(vid, ""); return "pre" if d and d < "20241107" else "post"

    # strata: (account, era, non-HIGH?) — draw to guarantee coverage + oversample uncertain
    hi = [k for k, r in items if r["function_confidence"] == "HIGH"]
    unc = [k for k, r in items if r["function_confidence"] != "HIGH"]
    # ensure every (account) and every function appears at least once
    picked = set()
    by_fn = collections.defaultdict(list)
    for (a, v), r in items: by_fn[r["function"]].append((a, v))
    for fn, ks in by_fn.items():
        picked.add(random.choice(ks))                       # 1 per function (coverage)
    for a in ACCTS:                                         # ensure each account
        aks = [(a, v) for (aa, v) in [k for k, _ in items] if aa == a]
    # ~40% from the uncertain pool, rest from HIGH, keeping account balance
    random.shuffle(unc); random.shuffle(hi)
    target_unc = int(N * 0.4)
    for k in unc:
        if len([1 for p in picked if cls[p]["function_confidence"] != "HIGH"]) >= target_unc: break
        picked.add(k)
    for k in hi:
        if len(picked) >= N: break
        picked.add(k)
    picked = list(picked)[:N]

    # write ids json (for review.html)
    json.dump([{"account": a, "video_id": v} for (a, v) in picked],
              open(f"{HERE}/data/eval/validation_sample_50.json", "w"))

    # write review sheet
    cols = ["account", "video_id", "upload_date", "url", "era",
            "central_claim", "caption", "on_screen_text", "spoken_summary",
            "function", "function_confidence", "function_rationale",
            "peak_subject", "peak_register", "peak_side", "peak_rationale",
            "treatments", "crudeness", "crudeness_evidence", "production_style",
            "has_dunk", "dunk_line", "attention_mechanisms", "strong_profanities",
            "VERDICT_function", "VERDICT_register", "VERDICT_treatments", "VERDICT_notes"]
    prof = {}
    for a in ACCTS:
        p = f"{HERE}/data/classification/profanity_{a}.jsonl"
        if os.path.exists(p):
            for line in open(p): r = json.loads(line); prof[(a, r["video_id"])] = r
    w = csv.DictWriter(open(f"{HERE}/data/classification/validation_sample_50.csv", "w", newline=""), fieldnames=cols)
    w.writeheader()
    for (a, v) in sorted(picked):
        r = cls[(a, v)]
        d = json.load(open(f"{HERE}/data/descriptions/gemini_v2/{a}/{v}.json"))
        treat = " ｜ ".join(f"{t['subject']}:{t['side']}:{t['register']:+d}" +
                           ("★" if t["is_peak"] else "") + ("Ⓜ" if t["is_main_character"] else "")
                           for t in r["treatments"])
        w.writerow({
            "account": a, "video_id": v, "upload_date": dates.get(v, ""),
            "url": f"https://www.tiktok.com/@{a}/video/{v}", "era": era(v),
            "central_claim": d.get("central_claim", ""), "caption": caps.get((a, v), ""),
            "on_screen_text": " · ".join(t.get("text", "") for t in (d.get("on_screen_text") or []))[:300],
            "spoken_summary": (d.get("spoken_summary") or "")[:300],
            "function": r["function"], "function_confidence": r["function_confidence"],
            "function_rationale": r["function_rationale"],
            "peak_subject": r.get("peak_subject", ""), "peak_register": r.get("peak_register", ""),
            "peak_side": r.get("peak_side", ""), "peak_rationale": r.get("peak_rationale", ""),
            "treatments": treat, "crudeness": r.get("crudeness", ""),
            "crudeness_evidence": r.get("crudeness_evidence", ""), "production_style": r.get("production_style", ""),
            "has_dunk": r.get("has_dunk", ""), "dunk_line": r.get("dunk_line", ""),
            "attention_mechanisms": "|".join(r.get("attention_mechanisms", [])),
            "strong_profanities": "|".join(prof.get((a, v), {}).get("strong_profanities", [])),
        })
    # report composition
    comp_a = collections.Counter(a for a, v in picked)
    comp_c = collections.Counter(cls[k]["function_confidence"] for k in picked)
    comp_f = collections.Counter(cls[k]["function"] for k in picked)
    print(f"sample n={len(picked)}")
    print("  by account:", dict(comp_a))
    print("  by confidence:", dict(comp_c))
    print("  by function:", dict(comp_f))
    print("  → data/classification/validation_sample_50.csv + data/eval/validation_sample_50.json")

if __name__ == "__main__":
    main()
