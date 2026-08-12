"""
transcribe_scope.py — layout-aware Whisper runner for the in-scope corpus.

The original transcribe.py only understands the party layout (videos/{account},
data/transcripts/{account}). Politician assets live under videos/politicians/{account}
and data/politicians/transcripts/{account} — which is why their transcripts never
got finished. This runner handles BOTH layouts, skips already-done files, and uses
the identical Whisper turbo invocation as transcribe.py.

Run ONE process at a time (orphan-process lesson). Kill before restarting.

Usage:
  python transcribe_scope.py                 # the 16 in-scope politicians (default)
  python transcribe_scope.py aoc tedcruz     # specific accounts (party or politician)
"""

import subprocess, sys, json
from pathlib import Path
from tqdm import tqdm

DATA = Path("data")
VIDEOS = Path("videos")
WHISPER_MODEL = "turbo"

PARTY = {"democrats", "dccc", "whitehouse", "senatedemshq", "senatedems", "republicans"}

# Corpus scope decided 2026-06-24 (see project memory).
SCOPE_POL = [
    "aoc", "bernie", "gavinnewsom", "petebuttigieg", "kamalaharris", "timwalz",
    "corybooker", "reprokhanna", "repmaxwellfrost", "realdonaldtrump", "jd",
    "vivekramaswamy", "tulsigabbard", "laratrump", "tedcruz", "senschumer",
]


def resolve(account):
    """Return (video_dir, transcript_dir) by detecting the account's layout."""
    if account in PARTY or (VIDEOS / account).is_dir():
        return VIDEOS / account, DATA / "transcripts" / account
    return VIDEOS / "politicians" / account, DATA / "politicians" / "transcripts" / account


def transcribe_file(video_path: Path, txt_path: Path):
    try:
        subprocess.run(
            ["whisper", str(video_path),
             "--model", WHISPER_MODEL,
             "--output_format", "json",
             "--output_dir", str(txt_path.parent),
             "--language", "en"],
            capture_output=True, text=True, timeout=300,
        )
        json_out = txt_path.parent / f"{video_path.stem}.json"
        if json_out.exists():
            text = json.loads(json_out.read_text()).get("text", "").strip()
            txt_path.write_text(text)
            json_out.unlink()
            return True
    except subprocess.TimeoutExpired:
        print(f"\n  timeout: {video_path.name}")
    except Exception as e:
        print(f"\n  error {video_path.name}: {e}")
    return False


def transcribe_account(account):
    vdir, tdir = resolve(account)
    if not vdir.is_dir():
        print(f"@{account}: no video dir at {vdir} — skipping")
        return
    tdir.mkdir(parents=True, exist_ok=True)
    videos = sorted(vdir.glob("*.mp4"))
    remaining = [v for v in videos if not (tdir / f"{v.stem}.txt").exists()]
    print(f"@{account}: {len(videos)} videos, {len(videos)-len(remaining)} done, "
          f"{len(remaining)} to transcribe")
    ok = 0
    for v in tqdm(remaining, desc=f"@{account}"):
        if transcribe_file(v, tdir / f"{v.stem}.txt"):
            ok += 1
    print(f"@{account}: transcribed {ok}/{len(remaining)}")


def main():
    accounts = sys.argv[1:] if len(sys.argv) > 1 else SCOPE_POL
    print(f"Transcribing {len(accounts)} account(s): {', '.join(accounts)}\n")
    for a in accounts:
        transcribe_account(a)
    print("\nDone. Re-run pipeline_status.py --scope to confirm coverage.")


if __name__ == "__main__":
    main()
