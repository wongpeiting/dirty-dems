"""
Music-ID timeline — AUTHORITATIVE audio track identification via ACRCloud fingerprinting.

The description LLM's song guesses aren't reliable (checked 2026-07-01); song identity comes
from here. Built as a TIMELINE, not one label, because:
  - music can CHANGE mid-post  -> we fingerprint multiple windows across the clip
  - a post can be SILENT then music enters -> ffmpeg silencedetect finds when sound starts

Also note: 91% of WH posts carry track="original sound" in TikTok metadata (the
broadcast-vs-participate signal), so the real song is hidden inside "original sound" and
only fingerprinting recovers it.

Setup:
  pip install pyacrcloud
  Add to .env:  ACRCLOUD_HOST, ACRCLOUD_ACCESS_KEY, ACRCLOUD_ACCESS_SECRET
  (free account at acrcloud.com -> create an "Audio & Video Recognition" project)

Usage:
  python music_id.py --test <video_id>            # one video, print timeline
  python music_id.py --manifest data/eval/gold_set.json
  python music_id.py --accounts whitehouse

Cost note: ACRCloud requests = windows. ~5 windows/video. 50 videos ≈ 250 requests (fine
on free tier); the full ~2,352 corpus ≈ ~12k requests — check your plan before rollout.
"""
import argparse, json, os, re, subprocess, time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import description_config as C

load_dotenv("../.env")
VIDEOS = C.HERE / "videos"
OUT = C.HERE / "data" / "music"
PARTY = ["whitehouse", "democrats", "republicans"]

WINDOW_SEC = 10          # length of each fingerprint sample
MAX_WINDOWS = 8          # cap requests per video (cost control)
SILENCE_DB = "-40dB"     # silencedetect threshold
SILENCE_MIN = 0.6        # min silence duration (s) to report
MIN_SLEEP = 1.0          # between ACRCloud calls (rate-limit courtesy)

_req_count = 0


def log(m): print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def video_path(account, vid):
    for name in (f"{vid}.mp4", f"{vid}_carousel.mp4"):
        p = VIDEOS / account / name
        if p.exists():
            return p
    return None


def duration_sec(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def silence_segments(path):
    """Return list of (start, end) silent spans via ffmpeg silencedetect (free/local)."""
    try:
        r = subprocess.run(["ffmpeg", "-i", str(path), "-af",
                            f"silencedetect=noise={SILENCE_DB}:d={SILENCE_MIN}", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=90)
        txt = r.stderr
    except Exception:
        return []
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", txt)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", txt)]
    return list(zip(starts, ends + [None] * (len(starts) - len(ends))))


def in_silence(t, sil):
    for s, e in sil:
        if s <= t and (e is None or t < e):
            return True
    return False


# --------------------------------------------------------------------- ACRCloud
_rec = None
def recognizer():
    global _rec
    if _rec is None:
        from acrcloud.recognizer import ACRCloudRecognizer
        cfg = {"host": os.getenv("ACRCLOUD_HOST"),
               "access_key": os.getenv("ACRCLOUD_ACCESS_KEY"),
               "access_secret": os.getenv("ACRCLOUD_ACCESS_SECRET"),
               "timeout": 12}
        if not all(cfg[k] for k in ("host", "access_key", "access_secret")):
            raise RuntimeError("ACRCloud keys missing in .env (ACRCLOUD_HOST/ACCESS_KEY/ACCESS_SECRET)")
        _rec = ACRCloudRecognizer(cfg)
    return _rec


class FatalACR(Exception):
    """ACRCloud auth/quota/limit error — stop the whole run (don't keep burning calls)."""


# 1001 = no match; 2xxx = per-FILE/window audio issues (no fingerprint, decode error, etc.)
# — all mean "skip this window, keep going". Only >= 3000 (auth/quota/signature) is fatal.


def identify_window(path, start):
    """Fingerprint a WINDOW_SEC sample starting at `start`. Returns a match dict, or None
    for a genuine no-result. Raises FatalACR on auth/quota/limit codes (>= 3000)."""
    global _req_count
    _req_count += 1
    res = recognizer().recognize_by_file(str(path), int(start), WINDOW_SEC)
    time.sleep(MIN_SLEEP)
    data = json.loads(res)
    code = data.get("status", {}).get("code")
    if code == 0:
        musics = data.get("metadata", {}).get("music", [])
        if not musics:
            return None
        m = musics[0]
        return {"track": m.get("title"),
                "artist": ", ".join(a.get("name", "") for a in m.get("artists", [])),
                "album": (m.get("album") or {}).get("name"),
                "score": m.get("score")}
    if code == 1001 or 2000 <= code < 3000:
        return None   # no match, or a per-file audio/decode issue — skip window, continue
    # >= 3000 = auth / signature / quota / limit — STOP so we don't runaway-bill or mislabel
    raise FatalACR(f"ACRCloud status {code}: {data.get('status', {}).get('msg')}")


def process(account, vid):
    path = video_path(account, vid)
    if not path:
        return None
    dur = duration_sec(path)
    sil = silence_segments(path)
    if dur <= 0:
        return None
    # window start times, capped, skipping windows centred in silence
    n = min(MAX_WINDOWS, max(1, int(dur // WINDOW_SEC) + 1))
    starts = [round(dur * i / n, 1) for i in range(n)]
    timeline = []
    for s in starts:
        if in_silence(s + WINDOW_SEC / 2, sil):
            timeline.append({"start": s, "status": "silence"})
            continue
        try:
            m = identify_window(path, s)
        except FatalACR:
            raise                     # propagate — stops the whole run
        except Exception as e:
            timeline.append({"start": s, "status": "error", "error": str(e)[:80]})
            continue
        if m:
            timeline.append({"start": s, "status": "music", **m})
        else:
            timeline.append({"start": s, "status": "unidentified"})
    tracks = sorted({t["track"] for t in timeline if t.get("track")})
    return {
        "video_id": vid, "account": account, "duration": round(dur, 1),
        "silent_intro": bool(sil and sil[0][0] < 1.0),
        "distinct_tracks": tracks,
        "track_changes": len(tracks) > 1,
        "timeline": timeline,
    }


def ids_from_manifest(p):
    by = {}
    for e in json.loads(Path(p).read_text()):
        by.setdefault(e["account"], []).append(e["video_id"])
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", nargs="*", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-requests", type=int, default=8000, help="hard cap on ACRCloud calls this run (resumable)")
    args = ap.parse_args()

    if args.test:
        for a in (args.accounts or PARTY):
            if video_path(a, args.test):
                print(json.dumps(process(a, args.test), ensure_ascii=False, indent=2))
                log(f"ACRCloud requests used: {_req_count}")
                return
        raise SystemExit(f"video {args.test} not found")

    if args.manifest:
        by = ids_from_manifest(args.manifest)
    else:
        by = {a: [p.stem.replace("_carousel", "") for p in (VIDEOS / a).glob("*.mp4")]
              for a in (args.accounts or PARTY)}
    for account, ids in by.items():
        outdir = OUT / account
        outdir.mkdir(parents=True, exist_ok=True)
        if args.limit:
            ids = ids[: args.limit]
        done = skip = 0
        for vid in ids:
            outp = outdir / f"{vid}.json"
            if outp.exists():
                skip += 1
                continue
            if _req_count >= args.max_requests:
                log(f"HIT --max-requests {args.max_requests} — stopping (resumable: re-run to continue). "
                    f"{_req_count} reqs used."); return
            try:
                r = process(account, vid)
            except FatalACR as e:
                log(f"FATAL ACRCloud error: {e}\n  STOP — likely quota/limit. Resumable: re-run to continue. "
                    f"({_req_count} reqs used, {done} written this run)"); return
            if r:
                outp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
                done += 1
                if done % 25 == 0:
                    log(f"  {account}: {done} done · {_req_count} ACRCloud reqs")
        log(f"{account}: wrote {done}, skipped {skip} · {_req_count} ACRCloud requests total")


if __name__ == "__main__":
    main()
