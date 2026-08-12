"""
Overlay-text pipeline — FULLY LOCAL, zero API cost. TEXT ONLY.

Extracts on-screen text from TikTok videos (captions, chyrons, meme text).

EMOJIS: a controlled test showed local emoji template-matching is unreliable
(true/false match scores overlap; the skull 💀 false-fires on dark regions). The
call (2026-06-29): capture emojis via the multimodal description layer (Gemini reads
them accurately) instead. The EmojiMatcher class is kept behind --emoji-experimental
as a rough hint only, NOT ground truth.

Stack (all already installed — NOT Tesseract):
  - PySceneDetect  -> representative frame per scene (cheaper than every-second)
  - cv2            -> frame grabbing
  - easyocr        -> modern OCR for the text layer

Output: data/ocr/{account}/{id}.json
  { video_id, account, source, n_frames, frame_times, overlay_text:[lines], full_text }

Usage:
  python ocr_overlays.py --test <video_id>        # one video, print result, don't save
  python ocr_overlays.py --accounts whitehouse    # a whole account
  python ocr_overlays.py                          # all 3 party coding-sheet accounts
  python ocr_overlays.py --ids <id> <id> ...      # just these video_ids (e.g. the pilot 10)
"""
import argparse, json, os, re, sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
VIDEOS = HERE / "videos"
OUT = HERE / "data" / "ocr"
PARTY = ["whitehouse", "democrats", "republicans"]
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"

# High-frequency reaction emojis the sentiment scale actually leans on.
EMOJI_SET = ["🔥", "😂", "😭", "💀", "🤣", "🇺🇸", "🦅", "😎", "🤭", "👏",
             "🤡", "💅", "🙄", "😀", "😞", "❤️", "💯", "🥀", "👀", "🫡",
             "🤬", "😱", "🙏", "🎉", "✨", "😈", "💪", "🚨", "⚠️", "👑"]

MAX_FRAMES = 12          # cap frames per video
SCENE_THRESHOLD = 27.0   # PySceneDetect ContentDetector sensitivity


# ----------------------------------------------------------------- frame sampling
def video_candidates(account, vid):
    """All existing video files for an id, largest first — a broken stub {id}.mp4
    can coexist with the real {id}_carousel.mp4, so we try the bigger one first."""
    cands = [VIDEOS / account / f"{vid}.mp4", VIDEOS / account / f"{vid}_carousel.mp4"]
    cands = [p for p in cands if p.exists()]
    return sorted(cands, key=lambda p: p.stat().st_size, reverse=True)


def find_video(account, vid):
    c = video_candidates(account, vid)
    return c[0] if c else None


def grab_frames(path, max_frames=MAX_FRAMES):
    """Pick up to max_frames visually-distinct frames using cv2 ONLY (no scenedetect/
    PyAV — that conflicts with cv2's bundled libav and segfaults in batch). We sample
    candidates, score change vs the previous candidate by downsized-gray mean-abs-diff,
    and keep the first frame + the biggest-change frames (where overlays change)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        # All-numeric TikTok filenames can trip cv2's image-sequence pattern parser
        # (icvExtractPattern) on very long IDs. Retry via a safe-named temp copy.
        import tempfile, shutil, os as _os
        tmp = Path(tempfile.gettempdir()) / f"ocr_safe_{_os.getpid()}.mp4"
        shutil.copy(path, tmp)
        cap = cv2.VideoCapture(str(tmp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    ncand = min(total, max(max_frames, max_frames * 4))
    cand, prev = [], None
    for k in range(ncand):
        i = int(total * k / ncand)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 64)).astype("int16")
        d = 0.0 if prev is None else float(np.mean(np.abs(small - prev)))
        cand.append((i, fr, d))
        prev = small
    cap.release()
    if not cand:
        return []
    keep = {cand[0][0]}                                  # always the first frame
    for i, _, _ in sorted(cand, key=lambda c: -c[2]):    # then biggest visual changes
        if len(keep) >= max_frames:
            break
        keep.add(i)
    return [(round(i / fps, 1), fr) for i, fr, _ in cand if i in keep]


# ------------------------------------------------------------------- text layer
_reader = None
def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def norm_line(s):
    return re.sub(r"\s+", " ", s).strip()


def ocr_text(frames):
    reader = get_reader()
    seen, lines = set(), []
    for _, fr in frames:
        for (_, txt, conf) in reader.readtext(fr):
            t = norm_line(txt)
            letters = sum(c.isalpha() for c in t)
            if conf < 0.45 or letters < 3:
                continue
            key = re.sub(r"[^a-z0-9]", "", t.lower())
            if key and key not in seen:
                seen.add(key)
                lines.append(t)
    return lines


# ------------------------------------------------------------------ emoji layer
class EmojiMatcher:
    """Match a curated emoji set by template-matching Apple-rendered glyphs
    (RGB template + alpha mask) at a few scales relative to frame height."""
    STRIKE = 160  # a valid Apple Color Emoji bitmap strike

    def __init__(self, emojis=EMOJI_SET):
        self.templates = {}
        if not os.path.exists(EMOJI_FONT):
            print("    (no Apple emoji font — emoji layer disabled)")
            return
        font = ImageFont.truetype(EMOJI_FONT, self.STRIKE)
        for e in emojis:
            img = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((10, 10), e, font=font, embedded_color=True)
            bbox = img.getbbox()
            if not bbox:
                continue
            glyph = img.crop(bbox)
            arr = np.array(glyph)  # RGBA
            self.templates[e] = arr

    def detect(self, frame_bgr, scales=(28, 38, 52, 70), max_sqdiff=0.18):
        """TM_SQDIFF_NORMED with the glyph's alpha as mask (0 = perfect match).
        A colour-variance gate at the matched location rejects flat-background
        false positives (emoji glyphs are colourful, not uniform)."""
        if not self.templates:
            return {}
        H, W = frame_bgr.shape[:2]
        found = {}
        for e, rgba in self.templates.items():
            best = 1.0
            best_loc = None
            best_s = 0
            for s in scales:
                if s >= H or s >= W:
                    continue
                g = cv2.resize(rgba, (s, s), interpolation=cv2.INTER_AREA)
                tmpl = cv2.cvtColor(g[:, :, :3], cv2.COLOR_RGB2BGR)
                mask = g[:, :, 3]
                if mask.max() == 0:
                    continue
                mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                try:
                    res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_SQDIFF_NORMED, mask=mask3)
                    res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
                    mn, _, loc, _ = cv2.minMaxLoc(res)
                    if mn < best:
                        best, best_loc, best_s = mn, loc, s
                except cv2.error:
                    continue
            if best_loc is None or best > max_sqdiff:
                continue
            # colour-variance gate: the matched patch must be colourful like a glyph
            x, y = best_loc
            patch = frame_bgr[y:y + best_s, x:x + best_s]
            if patch.size and patch.std() > 18:
                found[e] = round(1.0 - best, 3)  # report as a 0..1 confidence
        return found


# --------------------------------------------------------------------- per video
def process(account, vid, matcher):
    cands = video_candidates(account, vid)
    if not cands:
        return None
    frames, path = [], None
    for c in cands:               # fall back past a broken stub to the real file
        frames = grab_frames(c)
        if frames:
            path = c
            break
    if not frames:
        return None
    lines = ocr_text(frames)
    out = {
        "video_id": vid,
        "account": account,
        "source": path.name,
        "n_frames": len(frames),
        "frame_times": [t for t, _ in frames],
        "overlay_text": lines,
        "full_text": " ".join(lines),
    }
    if matcher:  # experimental, low-confidence hint only
        emojis = {}
        for _, fr in frames:
            for e, score in matcher.detect(fr).items():
                if e not in emojis or score > emojis[e]:
                    emojis[e] = score
        out["emojis_experimental"] = sorted(emojis, key=lambda e: -emojis[e])
    return out


def ids_for_account(account):
    sheet = HERE / f"{account}_coding_sheet.csv"
    if sheet.exists():
        import csv
        return [r["video_id"] for r in csv.DictReader(open(sheet, encoding="utf-8"))]
    # else: every video on disk
    d = VIDEOS / account
    return [p.stem.replace("_carousel", "") for p in d.glob("*.mp4")] if d.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", nargs="*", default=None)
    ap.add_argument("--ids", nargs="*", default=None, help="only these video_ids (e.g. the pilot 10)")
    ap.add_argument("--manifest", default=None, help="json [{account,video_id},...] to OCR (avoids huge shell arg lists)")
    ap.add_argument("--ids-file", default=None, help="file of whitespace/newline-separated video_ids")
    ap.add_argument("--test", default=None, help="one video_id: print, don't save")
    ap.add_argument("--emoji-experimental", action="store_true", help="add unreliable emoji hints")
    ap.add_argument("--limit", type=int, default=0, help="cap videos per account (debug)")
    args = ap.parse_args()

    matcher = EmojiMatcher() if args.emoji_experimental else None

    if args.test:
        for account in (args.accounts or PARTY):
            if find_video(account, args.test):
                r = process(account, args.test, matcher)
                print(json.dumps(r, ensure_ascii=False, indent=2))
                return
        sys.exit(f"video {args.test} not found in {args.accounts or PARTY}")

    by_acct = None
    if args.manifest:
        by_acct = {}
        for e in json.loads(Path(args.manifest).read_text()):
            by_acct.setdefault(e["account"], []).append(e["video_id"])
        accounts = list(by_acct)
        id_filter = None
    else:
        accounts = args.accounts or PARTY
        ids_arg = args.ids
        if args.ids_file:
            ids_arg = (ids_arg or []) + Path(args.ids_file).read_text().split()
        id_filter = list(dict.fromkeys(ids_arg)) if ids_arg else None

    for account in accounts:
        outdir = OUT / account
        outdir.mkdir(parents=True, exist_ok=True)
        if by_acct is not None:
            ids = [v for v in by_acct[account] if video_candidates(account, v)]
        elif id_filter:
            # process requested ids directly (any id with a video on disk),
            # NOT gated by coding-sheet membership — gold ids come from the WH-600 file
            ids = [v for v in id_filter if video_candidates(account, v)]
        else:
            ids = ids_for_account(account)
        if args.limit:
            ids = ids[: args.limit]
        done = skip = 0
        for vid in ids:
            outp = outdir / f"{vid}.json"
            if outp.exists():
                skip += 1
                continue
            r = process(account, vid, matcher)
            if r:
                outp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
                done += 1
                if done % 10 == 0:
                    print(f"  {account}: {done} done ({skip} skipped)")
        print(f"{account}: wrote {done}, skipped {skip} (already done)")


if __name__ == "__main__":
    main()
