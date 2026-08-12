"""
Phase 2 — description-layer evaluation, anchored on the WH-600 hand-coding (gold).

Two modes:
  --gold   score against data/eval/gold_set.json (the gold labels) — the VERIFIABLE eval:
             manipulation_correct  model.is_manipulated  vs  your fictional_overlay/ai_or_cgi/deep_fried tags
             cast_recall           model cast            vs  names in your notes
             faithfulness_vs_notes model description     vs  your notes (LLM-judge, your notes = reference)
             + schema_valid, completeness, onscreen_text_recall (vs local OCR)
  (default) reference-free pilot eval (vision judge), for non-gold videos.

Prints an A/B table (gemini vs claude) + a manipulation confusion matrix / P-R-F1 per arm,
and logs experiments to BrainTrust. Tune on the 50; then test the winning arm on the held-out 550.

Usage:
  python eval_descriptions.py --gold                 # the gold-anchored A/B
  python eval_descriptions.py --gold --no-braintrust
  python eval_descriptions.py --gold --engines gemini
"""
import argparse, json, re
from pathlib import Path
from dotenv import load_dotenv

import description_config as C
from description_schema import VideoDescription

load_dotenv("../.env")
JUDGE_MODEL = "claude-haiku-4-5-20251001"
GOLD_SET = C.HERE / "data" / "eval" / "gold_set.json"

# canonical surname -> aliases, for matching cast against your notes
CAST_VOCAB = {
    "trump": ["trump", "donald j. trump", "donald trump"], "vance": ["vance", "jd vance"],
    "musk": ["musk", "elon"], "jeffries": ["jeffries", "hakeem"], "schumer": ["schumer", "chuck schumer"],
    "obama": ["obama"], "biden": ["biden"], "harris": ["harris", "kamala"], "melania": ["melania"],
    "kirk": ["charlie kirk", "kirk"], "mamdani": ["mamdani"], "aoc": ["ocasio", "aoc"],
    "hegseth": ["hegseth"], "rubio": ["rubio"], "leavitt": ["leavitt"], "pelosi": ["pelosi"],
    "newsom": ["newsom"], "walz": ["walz"],
}


def names_in(text):
    t = (text or "").lower()
    return {canon for canon, aliases in CAST_VOCAB.items() if any(a in t for a in aliases)}


def toks(s):
    return set(re.findall(r"[a-z0-9]{3,}", (s or "").lower()))


# ------------------------------------------------------------------- data / io
def records(gold):
    if gold:
        return [{"account": r["account"], "video_id": r["video_id"], "gold": r}
                for r in json.loads(GOLD_SET.read_text())]
    pilot = json.loads(C.PILOT_MANIFEST.read_text())
    return [{"account": p["account"], "video_id": p["video_id"], "gold": None} for p in pilot]


def coding_row(account, vid):
    import csv
    f = C.HERE / f"{account}_coding_sheet.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["video_id"] == vid:
                return r
    return {}


def ocr_text(account, vid):
    p = C.OCR_DIR / account / f"{vid}.json"
    return json.loads(p.read_text()).get("full_text", "") if p.exists() else ""


def load_desc(engine, account, vid):
    p = C.DESC_DIR / engine / account / f"{vid}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------- deterministic
def s_schema_valid(d):
    try:
        o = VideoDescription.model_validate(d)
        return 1.0 if (o.visual_action.strip() and o.central_claim.strip()) else 0.5
    except Exception:
        return 0.0


def s_completeness(d):
    keys = ["visual_action", "cast", "on_screen_text", "spoken_summary",
            "central_claim", "intended_effect", "emotions_signaled"]
    return round(sum(1 for k in keys if d.get(k)) / len(keys), 3)


def s_onscreen_recall(d, account, vid):
    ref = toks(ocr_text(account, vid))
    if not ref:
        return None
    got = toks(" ".join(t.get("text", "") for t in d.get("on_screen_text", [])))
    return round(len(ref & got) / len(ref), 3)


def s_manipulation(d, gold):
    return 1.0 if bool(d.get("is_manipulated")) == bool(gold["is_manipulated_gold"]) else 0.0


def s_cast_recall(d, gold):
    g = names_in(gold.get("notes", "") + " " + gold.get("caption", ""))
    if not g:
        return None
    got = names_in(" ".join(c.get("name", "") for c in d.get("cast", [])))
    return round(len(g & got) / len(g), 3)


# ------------------------------------------------------------------- LLM judge
_judge = None
def judge_client():
    global _judge
    if _judge is None:
        import anthropic
        _judge = anthropic.Anthropic()
    return _judge


_CACHE = {}
def s_faithfulness_vs_notes(d, gold):
    """Your notes are the visual ground truth; judge whether the model's description
    matches them without contradiction. 0..1."""
    key = (gold["video_id"], id(d))
    if key in _CACHE:
        return _CACHE[key]
    prompt = (
        "A journalist hand-wrote NOTES describing what is in a political TikTok (ground truth). "
        "An AI separately produced a DESCRIPTION of the same video. Rate 0.0-1.0 how well the AI "
        "description matches the journalist's notes: 1.0 = captures the same key content/intent with "
        "no contradiction; low = misses the main point, contradicts, or invents things not in the notes.\n\n"
        f"JOURNALIST NOTES (ground truth): {gold.get('notes','')}\n"
        f"(your tags: {gold.get('tags','')}; subject: {gold.get('subject','')})\n\n"
        f"AI DESCRIPTION:\n central_claim: {d.get('central_claim')}\n"
        f" visual_action: {d.get('visual_action')}\n"
        f" is_manipulated: {d.get('is_manipulated')} — {d.get('manipulation_or_satire')}\n\n"
        'Return ONLY JSON: {"score": <float>, "notes": "<short why>"}'
    )
    msg = judge_client().messages.create(model=JUDGE_MODEL, max_tokens=250,
        messages=[{"role": "user", "content": prompt}])
    m = re.search(r"\{.*\}", msg.content[0].text, re.S)
    res = json.loads(m.group(0)) if m else {"score": None, "notes": "parse-fail"}
    _CACHE[key] = res
    return res


# ----------------------------------------------------------------------- run
def evaluate(engine, gold, use_judge):
    rows, conf = [], {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for rec in records(gold):
        account, vid, g = rec["account"], rec["video_id"], rec["gold"]
        d = load_desc(engine, account, vid)
        if d is None:
            continue
        sc = {"schema_valid": s_schema_valid(d), "completeness": s_completeness(d),
              "onscreen_text_recall": s_onscreen_recall(d, account, vid)}
        if g:
            sc["manipulation_correct"] = s_manipulation(d, g)
            sc["cast_recall"] = s_cast_recall(d, g)
            gm, mm = bool(g["is_manipulated_gold"]), bool(d.get("is_manipulated"))
            conf["TP" if gm and mm else "FN" if gm else "FP" if mm else "TN"] += 1
            if use_judge:
                j = s_faithfulness_vs_notes(d, g)
                sc["faithfulness_vs_notes"] = j.get("score")
                sc["_notes"] = j.get("notes", "")
        rows.append({"video_id": vid, "account": account, "gold": g, "scores": sc, "desc": d})
    return rows, conf


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def prf(c):
    tp, fp, fn = c["TP"], c["FP"], c["FN"]
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else None
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--gold-file", default=None, help="override the gold set json (e.g. holdout_set.json)")
    ap.add_argument("--engines", nargs="*", default=["gemini", "claude"])
    ap.add_argument("--no-braintrust", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    if args.gold_file:
        global GOLD_SET
        GOLD_SET = Path(args.gold_file)

    METRICS = ["schema_valid", "completeness", "onscreen_text_recall",
               "manipulation_correct", "cast_recall", "faithfulness_vs_notes"]
    results, confs = {}, {}
    for engine in args.engines:
        rows, conf = evaluate(engine, args.gold, use_judge=not args.no_judge)
        results[engine], confs[engine] = rows, conf
        if not args.no_braintrust and rows:
            try:
                import braintrust
                exp = braintrust.init(project="polititok-descriptions",
                                      experiment=engine + ("-gold" if args.gold else ""))
                for r in rows:
                    scores = {k: v for k, v in r["scores"].items()
                              if not k.startswith("_") and isinstance(v, (int, float))}
                    exp.log(input={"video_id": r["video_id"]}, output=r["desc"],
                            scores=scores, metadata={"engine": engine,
                            "gold_manip": (r["gold"] or {}).get("is_manipulated_gold")})
                exp.summarize()
            except Exception as e:
                print(f"  (BrainTrust skipped: {e})")

    print("\n=== A/B comparison (mean over set) ===")
    hdr = f"{'metric':24}" + "".join(f"{e:>12}" for e in args.engines)
    print(hdr); print("-" * len(hdr))
    for m in METRICS:
        line = f"{m:24}" + "".join(f"{str(mean([r['scores'].get(m) for r in results[e]])):>12}" for e in args.engines)
        print(line)
    print("\n(n per engine:", {e: len(results[e]) for e in args.engines}, ")")

    if args.gold:
        print("\n=== MANIPULATION detection vs your labels ===")
        for e in args.engines:
            c = confs[e]; p, r, f1 = prf(c)
            print(f"  {e:8} TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}  "
                  f"precision={p} recall={r} F1={f1}")
        print("\n=== manipulation DISAGREEMENTS (model vs your gold) — review these ===")
        for e in args.engines:
            for r in results[e]:
                g = r["gold"]
                if g and r["scores"].get("manipulation_correct") == 0.0:
                    gm = g["is_manipulated_gold"]; mm = r["desc"].get("is_manipulated")
                    print(f"  {e:7} {r['video_id']}  gold_manip={gm} model={mm} | {g.get('notes','')[:70]}")


if __name__ == "__main__":
    main()
