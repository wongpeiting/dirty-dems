"""
Deterministic profanity detection with EXACT citations — verbatim sources only
(raw Whisper transcript + OCR full_text + caption), NOT the Stage-1 summaries
(which paraphrase swearing away). Local, free, auditable.

Output: data/classification/profanity_{account}.jsonl
  {account, video_id, profanities: [exact matched strings], sources: {transcript|ocr|caption: [...]},
   strong: bool, n: int}
Tiers: STRONG (unambiguous swears incl. masked f---/f*ck forms) vs MILD (damn/hell/crap)
— reported separately so the WaPo-claim curve can use either bar.
"""
import csv, glob, json, os, re
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "data/classification"
OUT.mkdir(exist_ok=True)
ACCTS = ["whitehouse", "democrats", "republicans"]

STRONG = [
    r"\bf+u+c+k\w*", r"\bf[\*\-@#]{1,3}k\w*", r"\bf-{2,}\w*", r"\bfck\w*", r"\beffing\b",
    # bleeped/censored forms (Whisper + captions write these): f***, f***ing, f**, mother f***, s***
    r"\bf\*{1,4}\w*", r"\bmother\s?f[\*\w]{1,6}", r"\bs\*{2,4}t?\w*", r"\bb\*{1,4}ch\w*",
    # euphemized: "F-ed around", "f'd", "effing"/"effed" (NOT eff\w* — matches 'effect(s)'/'efforts')
    r"\bf[-']e?d\b(?=\s+(around|up|over|with))", r"\beffing\b", r"\beffed\b",
    # acronym expletives (the gold set counts these)
    r"\bfafo\b", r"\blfg\b", r"\bstfu\b", r"\bgtfo\b", r"\bwtf\b", r"\blmfao\b",
    r"\bmother\s?f\w*er\w*", r"\bsh[i1]t\w*", r"\bsh[\*\-@#]{1,3}t?\b", r"\bs-{2,}\b", r"\bbullsh\w*",
    r"\bdipsh\w*", r"\bass+hole\w*", r"\bdumb\s?ass\w*", r"\bjack\s?ass\w*", r"\bbad\s?ass\w*",
    r"\bbitch\w*", r"\bb[\*\-@#]{1,3}ch\w*", r"\bbastard\w*", r"\bgod\s?damn\w*",
    r"\bpiss\w*", r"\b[Dd][i1]ck(head)?\b(?!\s+[A-Z])", r"\bprick\b", r"\bcunt\w*", r"\bdouche\w*",
]
MILD = [r"\bdamn\w*", r"\bhell\b", r"\bcrap(py|s)?\b", r"\bsucks?\b", r"\bpissed\s?off\b"]  # crap(py|s) not crap\w* — Sen. Crapo
STRONG_RE = [re.compile(p, re.I) for p in STRONG]
MILD_RE = [re.compile(p, re.I) for p in MILD]
# meta-mentions: the Stage-1 description SAYS profanity occurs without quoting it
# NOTE: no bare 'cursed' (internet slang for weird imagery) — only active swearing verbs
META_RE = re.compile(r"bleep\w*|expletiv\w*|profanit\w*|profanely|censored (word|audio|language)|curses at|cursing|swear(s|ing| word)|f-bomb|vulgar language", re.I)


def scan(text):
    s, m = [], []
    if not text:
        return s, m
    for rx in STRONG_RE:
        s += rx.findall(text) if rx.groups == 0 else [x.group(0) for x in rx.finditer(text)]
    for rx in MILD_RE:
        m += rx.findall(text) if rx.groups == 0 else [x.group(0) for x in rx.finditer(text)]
    return s, m


def captions_for(a):
    caps = {}
    p = HERE / f"data/{a}_posts.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            caps[str(r["id"])] = r.get("title") or r.get("caption") or ""
    cs = HERE / f"{a}_coding_sheet.csv"
    if cs.exists():
        for r in csv.DictReader(open(cs)):
            if (r.get("caption") or "").strip():
                caps[r["video_id"]] = r["caption"]
    return caps


def main():
    grand = {}
    for a in ACCTS:
        caps = captions_for(a)
        rows, n_strong, n_any = [], 0, 0
        # scan every post that has ANY verbatim source
        vids = set(caps)
        vids |= {os.path.basename(p).replace("_carousel", "")[:-4] for p in glob.glob(str(HERE / f"data/transcripts/{a}/*.txt"))}
        for vid in sorted(vids):
            srcs, all_s, all_m = {}, [], []
            tp = glob.glob(str(HERE / f"data/transcripts/{a}/{vid}*.txt"))
            if tp:
                s, m = scan(open(tp[0], errors="ignore").read())
                if s or m: srcs["transcript"] = s + m
                all_s += s; all_m += m
            op = HERE / f"data/ocr/{a}/{vid}.json"
            if op.exists():
                try: otext = json.load(open(op)).get("full_text") or ""
                except Exception: otext = ""
                s, m = scan(otext)
                if s or m: srcs["ocr"] = s + m
                all_s += s; all_m += m
            s, m = scan(caps.get(vid, ""))
            if s or m: srcs["caption"] = s + m
            all_s += s; all_m += m
            # Stage-1 description: spoken_summary, visual_action, on_screen_text, audio_music
            dp = HERE / f"data/descriptions/gemini_v2/{a}/{vid}.json"
            meta_hit = False
            if dp.exists():
                try:
                    dd = json.load(open(dp))
                    dtext = " ".join([dd.get("spoken_summary") or "", dd.get("visual_action") or "",
                                      dd.get("audio_music") or "",
                                      " ".join(t.get("text", "") for t in (dd.get("on_screen_text") or []))])
                    s, m = scan(dtext)
                    if s or m: srcs["description"] = s + m
                    all_s += s; all_m += m
                    if META_RE.search(dtext):
                        meta_hit = True; srcs.setdefault("description_meta", []).append(META_RE.search(dtext).group(0))
                except Exception:
                    pass
            if all_s or all_m or meta_hit:
                n_any += 1
                if all_s or meta_hit: n_strong += 1
                rows.append({"account": a, "video_id": vid,
                             "profanities": sorted(set(w.lower() for w in all_s + all_m)),
                             "strong_profanities": sorted(set(w.lower() for w in all_s)),
                             "described_bleep": meta_hit,
                             "sources": srcs, "strong": bool(all_s) or meta_hit, "n": len(all_s) + len(all_m)})
        with open(OUT / f"profanity_{a}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        grand[a] = (len(vids), n_any, n_strong)
        print(f"{a}: scanned {len(vids)} posts · any profanity {n_any} ({100*n_any/max(1,len(vids)):.1f}%) · STRONG {n_strong} ({100*n_strong/max(1,len(vids)):.1f}%)")
    print("\nsample strong hits:")
    for a in ACCTS:
        for line in open(OUT / f"profanity_{a}.jsonl"):
            r = json.loads(line)
            if r["strong"]:
                print(f"  {a}/{r['video_id']}: {r['strong_profanities']} via {list(r['sources'])}")
                break


if __name__ == "__main__":
    main()
