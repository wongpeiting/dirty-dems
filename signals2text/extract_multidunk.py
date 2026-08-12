"""
Scoped dunk EXTRACTION over all has_dunk videos: pulls ALL distinct dunk lines per
post (the Stage-2 classifier kept only one), and DECONFLICTS song/sound-based lines
(a lyric/trending-sound/TV-skit line counts as a dunk ONLY if the video aims it at a
target; ambient music does not). Judged with the Stage-1 multimodal description in
context. Every dunk carries a `source` (caption/overlay/speech/sound) + `target`.
Full methodology: METHODOLOGY_DUNKS.md. Safety rig cloned from classify_function.py.

  python extract_multidunk.py --candidates data/analysis/test10_candidates.csv --limit 10 --cost-limit 0.60  # test
  python extract_multidunk.py --candidates data/analysis/all_dunk_candidates.csv --cost-limit 8.00           # full
Output: data/analysis/multidunk_extracted.jsonl (resumable; skips ids already done)
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("../.env")
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal

HERE = Path(__file__).parent
CAND = HERE / "data/analysis/all_dunk_candidates.csv"
OUT  = HERE / "data/analysis/multidunk_extracted.jsonl"
DESC = HERE / "data/descriptions/gemini_v2"
MODEL = "gemini-2.5-pro"
MIN_SLEEP = 3.0
RATE_IN, RATE_OUT = 1.25 / 1e6, 10.0 / 1e6
BILLING = ["billing", "quota", "credit", "resource_exhausted", "prepayment"]

class Dunk(BaseModel):
    line: str = Field(description="the dunk line, verbatim from the post")
    target: str = Field(description="who or what the dunk is aimed at; '' if it targets no one")
    source: Literal["caption", "overlay", "speech", "sound"] = Field(
        description="where the line comes from: caption text; overlay = on-screen text/graphic the "
                    "account added; speech = spoken narration; sound = a song / trending-audio / TV-skit "
                    "line used from the audio track")
class DunkList(BaseModel):
    dunk_lines: list[Dunk] = Field(description="every distinct dunk in the post; [] if none survive the bar")

CAP = {}
for a in ["democrats", "republicans", "whitehouse"]:
    for r in csv.DictReader(open(HERE / f"data/{a}_posts.csv")):
        CAP[str(r["id"])] = r.get("title") or ""

def first_exist(*paths):
    for p in paths:
        if p.exists(): return p
    return None

def load_inputs(a, vid):
    dp = first_exist(DESC/a/f"{vid}.json", DESC/a/f"{vid}_carousel.json")
    desc = json.load(open(dp)) if dp else {}
    op = first_exist(HERE/"data/ocr"/a/f"{vid}.json", HERE/"data/ocr"/a/f"{vid}_carousel.json")
    ocr = (json.load(open(op)).get("full_text", "") if op else "")
    tp = first_exist(HERE/"data/transcripts"/a/f"{vid}.txt", HERE/"data/transcripts"/a/f"{vid}_carousel.txt")
    tr = (open(tp, errors="ignore").read() if tp else "")
    mp = first_exist(HERE/"data/music"/a/f"{vid}.json", HERE/"data/music"/a/f"{vid}_carousel.json")
    track = ""
    if mp:
        try: track = "; ".join(t for t in (json.load(open(mp)).get("distinct_tracks") or []) if t)
        except: pass
    ost = " | ".join(x.get("text", "") for x in (desc.get("on_screen_text") or []) if isinstance(x, dict))
    return CAP.get(vid, ""), desc, ocr, tr, ost, track

PROMPT = """This official U.S. political-party TikTok post may dunk on one or more targets. Extract EVERY \
distinct DUNK — a short, witty put-down, burn, roast, mocking nickname, or taunt — VERBATIM where possible, \
one entry per distinct dunk. For each, give the line, who/what it targets, and its SOURCE.

CRITICAL — deconflicting song/sound audio:
- A song/sound is playing on this post: "{track}". Song lyrics are usually just ambience and are NOT dunks.
- BUT political TikTok often WEAPONIZES a sound — a trending song, TV-skit line, or lyric cut over footage of a \
target so the audio IS the burn. Then it counts.
- RULE: a line from a song / trending sound / TV skit is a dunk ONLY IF the video aims it at a target (the \
description/visuals pair it with someone being mocked). If the music is ambient with no target, it is NOT a dunk. \
Use "WHAT THE VIDEO DOES" below to decide intent and target.
- SOURCE tag per line: caption | overlay (on-screen text/graphic) | speech (spoken narration) | sound (from the song/trending-audio/skit).

EXCLUDE: app/UI text (Followers, Poll Tracker, timestamps), bare usernames/handles, hashtags alone, neutral \
narration, and plain policy statements with no barb.

WHAT THE VIDEO DOES (multimodal description — the ground truth for intent/target):
  action: {va}
  central claim: {cc}
  intended effect: {ie}
CLASSIFIER-FOUND TARGETS (opponents mocked): {targets}
CAPTION (account's own words): {cap}
ON-SCREEN TEXT (curated overlays the account added): {ost}
OCR (all on-screen text — noisy, filter hard): {ocr}
TRANSCRIPT (spoken OR SUNG — may be background-song lyrics, judge with the description): {tr}
"""

def build(row):
    a, vid = row["account"], row["video_id"]
    cap, desc, ocr, tr, ost, track = load_inputs(a, vid)
    return PROMPT.format(
        track=(track or "none identified")[:120],
        va=(desc.get("visual_action", "") or "")[:900],
        cc=(desc.get("central_claim", "") or "")[:400],
        ie=(desc.get("intended_effect", "") or "")[:400],
        targets=(row.get("opponent_targets", "") or "")[:300],
        cap=cap[:600], ost=ost[:1200], ocr=ocr[:2200], tr=tr[:2200])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cost-limit", type=float, default=8.0)
    ap.add_argument("--candidates", default=str(CAND))
    args = ap.parse_args()

    cands = list(csv.DictReader(open(args.candidates)))
    done = {json.loads(l)["video_id"] for l in open(OUT)} if OUT.exists() else set()
    todo = [c for c in cands if c["video_id"] not in done]
    if args.limit: todo = todo[:args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] candidates={len(cands)} done={len(done)} todo={len(todo)} "
          f"model={MODEL} cost_limit=${args.cost_limit}", flush=True)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    fout = open(OUT, "a")
    total_cost, consec, n = 0.0, 0, 0
    for c in todo:
        vid, a = c["video_id"], c["account"]
        if total_cost >= args.cost_limit:
            print(f"COST LIMIT ${args.cost_limit} reached — stopping", flush=True); break
        t = time.time()
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=build(c),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=DunkList))
            u = resp.usage_metadata
            cost = (u.prompt_token_count or 0)*RATE_IN + (u.candidates_token_count or 0)*RATE_OUT
            total_cost += cost
            dl = [{"line": d.line, "target": d.target, "source": d.source} for d in resp.parsed.dunk_lines]
            row = {"video_id": vid, "account": a, "n_opponents_mocked": int(c["n_opponents_mocked"]),
                   "llm_dunk_line": c["llm_dunk_line"], "url": c["url"], "dunk_lines": dl}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
            n += 1; consec = 0
            nsound = sum(1 for d in dl if d["source"] == "sound")
            print(f"[{time.strftime('%H:%M:%S')}] ✓ {vid} @{a[:4]} -> {len(dl)} dunks ({nsound} sound) "
                  f"${cost:.4f} ({time.time()-t:.1f}s) total=${total_cost:.4f}", flush=True)
        except Exception as e:
            if any(k in str(e).lower() for k in BILLING):
                print(f"FATAL (billing) {vid}: {e}\n STOP.", flush=True); fout.close(); sys.exit(2)
            consec += 1
            print(f"  ✗ {vid}: {str(e)[:150]} (consec {consec})", flush=True)
            if consec >= 5:
                print("CIRCUIT BREAKER: 5 consecutive errors — stopping.", flush=True); fout.close(); sys.exit(3)
            time.sleep(min(60, 2**consec * 3)); continue
        el = time.time() - t
        if el < MIN_SLEEP: time.sleep(MIN_SLEEP - el)
    fout.close()
    print(f"[{time.strftime('%H:%M:%S')}] DONE: {n} videos, session cost ${total_cost:.4f}", flush=True)

if __name__ == "__main__":
    main()
