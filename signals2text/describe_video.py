"""
Description layer (Stage 1) — safe two-arm runner.

Turns each pilot TikTok into a structured, evidence-bearing VideoDescription, using
EITHER arm:
  Arm A (gemini) — native video -> Gemini, structured output via response_schema
  Arm B (claude) — scene keyframes + transcript/OCR -> Claude, structured via tool-use

Safety machinery cloned from embed_safe.py (the $907 lesson): single stream, 3s min
sleep, exponential backoff on ANY error, circuit breaker at 5 consecutive errors,
immediate stop on billing keywords, failed-id tracking, real-time cost, --dry-run,
--cost-limit. Test 1 video, read the cost, THEN do the rest.

Usage:
  python describe_video.py --engine gemini --limit 1            # cost test: 1 pilot video
  python describe_video.py --engine gemini                      # all pilot videos, Arm A
  python describe_video.py --engine claude                      # all pilot videos, Arm B
  python describe_video.py --engine gemini --ids <id> --dry-run # show plan, no API call
"""
import argparse, base64, json, os, sys, time, random
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import description_config as C
from description_schema import VideoDescription
import ocr_overlays as ocr  # reuse scene-frame sampling + video-candidate logic

load_dotenv("../.env")

FAILED_IDS, ATTEMPTED_IDS = set(), set()
_consecutive_errors = 0
_session_cost = 0.0


class BadVideo(Exception):
    """A per-video data problem (corrupt file, empty/safety-blocked output). Skip the video
    WITHOUT counting toward the circuit breaker — the breaker is for billing/API-health runaway."""


# API messages that mean 'this one video is bad', not 'the API is in trouble'
PER_VIDEO_ERR = ["invalid_argument", "corrupted", "0 frames", "wrong video metadata",
                 "json_type", "validation error for videodescription"]


def log(m): print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


# ----------------------------------------------------------------- input bundle
def load_pilot(manifest=None):
    return json.loads(Path(manifest).read_text() if manifest else C.PILOT_MANIFEST.read_text())


def coding_row(account, vid):
    import csv
    for r in csv.DictReader(open(C.HERE / f"{account}_coding_sheet.csv", encoding="utf-8")):
        if r["video_id"] == vid:
            return r
    return {}


_FACES = {}
def _load_faces(account):
    """video_id -> {person: best_tier} from pol-face presence_{account}.csv (read live)."""
    if account in _FACES:
        return _FACES[account]
    import csv
    m = {}
    p = C.FACE_DIR / f"presence_{account}.csv"
    if p.exists():
        for row in csv.DictReader(open(p, encoding="utf-8")):
            vid, person, tier = row["video_id"], row["person"], row.get("confidence_tier", "")
            d = m.setdefault(vid, {})
            if person not in d or tier == "high":
                d[person] = tier
    # union the human-confirmed manual silhouette/meme/composite additions (mostly Trump)
    mp = C.FACE_DIR / f"manual_appearances_{account}.csv"
    if mp.exists():
        for row in csv.DictReader(open(mp, encoding="utf-8")):
            person = (row.get("person") or "").strip()
            if not person:
                continue
            d = m.setdefault(row["video_id"], {})
            d.setdefault(person, "manual")   # don't override a detected tier
    _FACES[account] = m
    return m


def faces_for(account, vid):
    d = _load_faces(account).get(vid, {})
    pretty = lambda p: " ".join(w.capitalize() for w in p.split("-"))
    # high-confidence first
    return [pretty(p) + ("" if t == "high" else f" ({t})")
            for p, t in sorted(d.items(), key=lambda x: (x[1] != "high", x[0]))]


def music_summary(account, vid):
    p = C.MUSIC_DIR / account / f"{vid}.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    if not d.get("distinct_tracks"):
        return "no commercial song matched (likely original/meme audio or speech — describe what you hear)"
    seg = []
    for t in d.get("timeline", []):
        if t.get("status") == "music" and t.get("track"):
            seg.append(f"{t['start']}s: {t['track']}" + (f" — {t['artist']}" if t.get("artist") else ""))
        elif t.get("status") == "silence":
            seg.append(f"{t['start']}s: (silence)")
    return "; ".join(seg) + (" [track changes mid-post]" if d.get("track_changes") else "")


def context_text(account, vid):
    r = coding_row(account, vid)
    ocr_p = C.OCR_DIR / account / f"{vid}.json"
    ocr_text = json.loads(ocr_p.read_text()).get("full_text", "") if ocr_p.exists() else ""
    faces = faces_for(account, vid)
    return (
        "CONTEXT — trusted side-signals to GROUND your description (the video itself is primary truth):\n"
        f"- Account: @{account}\n"
        f"- Caption: {r.get('caption','')}\n"
        f"- TikTok sound label: {r.get('track','')}  (\"original sound\" = the account's OWN audio)\n"
        f"- Whisper transcript (audio→text): {r.get('transcript','')}\n"
        f"- On-screen text OCR (local, may be noisy): {ocr_text}\n"
        f"- AUTHORITATIVE face-ID (InsightFace vs a known-politician roster) — people confirmed on screen: "
        f"{', '.join(faces) if faces else '(none detected / not yet processed)'}\n"
        f"- AUTHORITATIVE music (ACRCloud fingerprint): {music_summary(account, vid) or '(not yet processed)'}\n"
    )


TODAY = datetime.now().strftime("%B %d, %Y")
PROMPT = (
    "You are reconstructing what a political TikTok COMMUNICATES, for a journalism "
    "dataset. These are REAL posts by official US political accounts from November 2024 "
    "through 2026 — they may reference events AFTER your knowledge cutoff. "
    f"For reference, TODAY'S DATE IS {TODAY}. Any date on or before today is REAL and in the "
    "past or present — a date being later than what your training knows is NEVER, by itself, "
    "evidence of fabrication. Do NOT set is_manipulated true just because a date looks like the "
    "future. Treat the video as ground truth about real events; only flag manipulation you can "
    "actually SEE (photoshop, superimposition, deepfake, AI-gen, staged fake). "
    "Observe the video (and the provided context) and produce a structured, "
    "evidence-bearing description per the schema. RULES: (1) Describe only — do NOT "
    "assign any sentiment, attack, or persuasion labels; that is a separate later step. "
    "(2) Preserve emojis verbatim in on_screen_text and emojis_present. (3) Ground every "
    "cast member and the central_claim in a concrete cue (what was shown/said/written). "
    "(4) If unsure, say so in uncertainty_flags rather than guessing. "
    "(5) ACTIVELY look for VISIBLE visual manipulation — photoshops, superimposed objects "
    "(e.g. a sombrero added onto a politician), deepfakes, AI-generated imagery, staged fake "
    "documents/broadcasts — made to mock or mislead. If present, set is_manipulated=true and "
    "describe it in manipulation_or_satire. This is DIFFERENT from the date guidance above: "
    "do not invent fakery from unfamiliar dates, but DO report manipulation you can actually SEE. "
    "(6) FACE-ID = PRESENCE ONLY, not main-character. The CONTEXT's face-ID list is people "
    "VISIBLY on screen (from face recognition) — include each in `cast` with how_identified='face'. "
    "But YOU decide `is_main`: the main character is who/what the post is centrally ABOUT. CRUCIAL: "
    "a person can be the main character even if their face NEVER appears — if the post is clearly "
    "about them (named, discussed, attacked, mocked, or implied), add them to `cast` "
    "(how_identified='named_in_caption'/'named_in_audio'/'on_screen_text'/'implied', "
    "role_in_post='spoken_about') and set is_main=true. Conversely, a face-ID'd person shown only "
    "as background/B-roll is NOT main. Presence (face-ID) and aboutness (main character) are "
    "different — judge aboutness from the whole post. "
    "The face-ID roster covers POLITICIANS ONLY, so 'none detected' does NOT mean no notable "
    "people. ACTIVELY identify famous NON-political people & fictional characters yourself "
    "(actors, musicians, athletes, cartoon characters) — a reaction-clip montage of Brad Pitt / "
    "Jim Carrey / SpongeBob, a Hannah Montana or Sabrina Carpenter meme, or a real visitor like "
    "Cristiano Ronaldo or a sports team. These are pop-culture BORROWING (a key tactic) or real "
    "cameos — put them in `cultural_references` (named), NEVER drop them. `cast` = political "
    "figures/orgs + the main character(s); `cultural_references` = the borrowed pop-culture. "
    "(7) TRUST the fingerprinted **music** in CONTEXT over your own ear for song identity. For "
    "sounds NOT fingerprinted — meme audio, sound effects, a comedic rimshot/'ba dum tss', "
    "record-scratch, sad-trombone — describe what you hear in audio_music; these baked-in meme "
    "sounds are important signal."
)


# ----------------------------------------------------------------------- cost
def record_cost(engine, usage_in, usage_out):
    global _session_cost
    rt = C.RATES[engine]
    cost = usage_in / 1e6 * rt["in"] + usage_out / 1e6 * rt["out"]
    _session_cost += cost
    return cost


# ------------------------------------------------------------------- Arm A: Gemini
_gemini = None
def gemini_client():
    global _gemini
    if _gemini is None:
        from google import genai
        _gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini


def gemini_describe(video_path, ctx, vid):
    from google.genai import types
    client = gemini_client()
    size = os.path.getsize(video_path)
    if size <= C.INLINE_MAX:
        with open(video_path, "rb") as f:
            vpart = types.Part(inline_data=types.Blob(mime_type="video/mp4", data=f.read()))
        contents = types.Content(parts=[vpart, types.Part(text=PROMPT + "\n\n" + ctx)])
        cleanup = None
    else:
        up = client.files.upload(file=str(video_path))
        while up.state.name == "PROCESSING":
            time.sleep(2); up = client.files.get(name=up.name)
        if up.state.name != "ACTIVE":
            raise RuntimeError("file upload not ACTIVE")
        contents = [up, PROMPT + "\n\n" + ctx]
        cleanup = up.name
    try:
        resp = client.models.generate_content(
            model=C.GEMINI_MODEL, contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=VideoDescription),
        )
    finally:
        if cleanup:
            try: client.files.delete(name=cleanup)
            except Exception: pass
    obj = resp.parsed
    if not isinstance(obj, VideoDescription):
        if not getattr(resp, "text", None):
            raise BadVideo(f"empty/blocked response (feedback: {getattr(resp, 'prompt_feedback', None)})")
        obj = VideoDescription.model_validate_json(resp.text)
    obj.video_id = vid
    u = resp.usage_metadata
    return obj, (u.prompt_token_count or 0), (u.candidates_token_count or 0)


# ------------------------------------------------------------------- Arm B: Claude
_claude = None
def claude_client():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _claude


def claude_describe(video_path, ctx, vid):
    import cv2
    client = claude_client()
    frames = ocr.grab_frames(video_path)[: C.CLAUDE_MAX_FRAMES]
    blocks = []
    for _, fr in frames:
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(buf).decode()}})
    tool = {"name": "emit_description",
            "description": "Return the structured video description.",
            "input_schema": VideoDescription.model_json_schema()}
    msg = client.messages.create(
        model=C.CLAUDE_MODEL, max_tokens=4000, tools=[tool],
        tool_choice={"type": "tool", "name": "emit_description"},
        messages=[{"role": "user", "content": [
            *blocks,
            {"type": "text", "text": PROMPT + "\n\n" + ctx +
             f"\n\n(These are {len(blocks)} scene keyframes sampled from the video.)"}]}])
    payload = next((b.input for b in msg.content if b.type == "tool_use"), None)
    if payload is None:
        raise RuntimeError("no tool_use block returned")
    payload["video_id"] = vid
    obj = VideoDescription.model_validate(payload)
    return obj, msg.usage.input_tokens, msg.usage.output_tokens


# --------------------------------------------------------------------- runner
def describe_one(engine, account, vid):
    cands = ocr.video_candidates(account, vid)
    if not cands:
        raise FileNotFoundError(f"no video for {account}/{vid}")
    ctx = context_text(account, vid)
    fn = gemini_describe if engine == "gemini" else claude_describe
    return fn(cands[0], ctx, vid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["gemini", "claude"], default="gemini",
                    help="gemini (chosen arm — native video, won the gold A/B). claude kept for comparison.")
    ap.add_argument("--manifest", default=None, help="path to a manifest json (default: pilot)")
    ap.add_argument("--out-tag", default=None, help="output subfolder under data/descriptions (default: engine name)")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cost-limit", type=float, default=C.COST_LIMIT_DEFAULT)
    args = ap.parse_args()
    global _consecutive_errors

    pilot = load_pilot(args.manifest)
    if args.ids:
        pilot = [p for p in pilot if p["video_id"] in set(args.ids)]
    if args.limit:
        pilot = pilot[: args.limit]

    outroot = C.DESC_DIR / (args.out_tag or args.engine)
    log(f"engine={args.engine} model={C.GEMINI_MODEL if args.engine=='gemini' else C.CLAUDE_MODEL} "
        f"videos={len(pilot)} cost_limit=${args.cost_limit}")
    if args.dry_run:
        for p in pilot:
            log(f"  would describe {p['account']}/{p['video_id']} ({p.get('reason','')})")
        return

    done = 0
    for p in pilot:
        account, vid = p["account"], p["video_id"]
        outp = outroot / account / f"{vid}.json"
        if outp.exists():
            log(f"  skip (done): {vid}"); continue
        if vid in ATTEMPTED_IDS:
            continue
        ATTEMPTED_IDS.add(vid)
        if _session_cost >= args.cost_limit:
            log(f"STOP: cost limit ${args.cost_limit} reached (spent ${_session_cost:.4f})"); break

        backoff = C.INITIAL_BACKOFF
        for attempt in range(4):
            try:
                t0 = time.time()
                obj, tin, tout = describe_one(args.engine, account, vid)
                cost = record_cost(args.engine, tin, tout)
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.write_text(obj.model_dump_json(indent=2), encoding="utf-8")
                done += 1; _consecutive_errors = 0
                log(f"  ✓ {vid}  in={tin} out={tout} tok  ${cost:.4f}  "
                    f"({time.time()-t0:.1f}s)  cast={len(obj.cast)}  total=${_session_cost:.4f}")
                break
            except BadVideo as e:
                log(f"  skip (bad video): {vid} — {str(e)[:90]}"); FAILED_IDS.add(vid); break
            except Exception as e:
                msg = str(e)
                if any(k.lower() in msg.lower() for k in C.FATAL_KEYWORDS):
                    log(f"FATAL (billing/quota) on {vid}: {msg}\n  STOP — check billing."); sys.exit(1)
                if any(k.lower() in msg.lower() for k in PER_VIDEO_ERR):
                    log(f"  skip (bad video): {vid} — {msg[:90]}"); FAILED_IDS.add(vid); break
                _consecutive_errors += 1
                log(f"  ! error on {vid} (attempt {attempt+1}): {msg[:160]}")
                if _consecutive_errors >= C.MAX_CONSECUTIVE_ERRORS:
                    log("CIRCUIT BREAKER: 5 consecutive errors — STOP. Check billing before rerun."); sys.exit(1)
                time.sleep(backoff); backoff = min(backoff * 2, C.MAX_BACKOFF)
            finally:
                time.sleep(random.uniform(C.MIN_SLEEP, C.MAX_SLEEP))
        else:
            FAILED_IDS.add(vid)

    log(f"done: {done} described · session cost ${_session_cost:.4f}"
        + (f" · failed: {sorted(FAILED_IDS)}" if FAILED_IDS else ""))


if __name__ == "__main__":
    main()
