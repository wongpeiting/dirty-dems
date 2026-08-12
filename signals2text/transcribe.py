"""
Transcribe TikTok videos using Whisper turbo.
Runs on all .mp4 files in videos/ subdirectories.
Skips files that already have a .txt transcript.

Usage:
  python transcribe.py                  # All accounts
  python transcribe.py democrats        # Specific account
"""

import subprocess, sys, json, time
from pathlib import Path
from tqdm import tqdm
from config import ACCOUNTS, VIDEOS_DIR, DATA_DIR

TRANSCRIPT_DIR = DATA_DIR / "transcripts"
WHISPER_MODEL = "turbo"


def transcribe_file(video_path: Path, transcript_path: Path) -> str | None:
    """Transcribe a single video file with Whisper."""
    try:
        result = subprocess.run(
            ["whisper", str(video_path),
             "--model", WHISPER_MODEL,
             "--output_format", "json",
             "--output_dir", str(transcript_path.parent),
             "--language", "en"],
            capture_output=True, text=True, timeout=300,
        )
        # Whisper outputs to {filename}.json
        json_out = transcript_path.parent / f"{video_path.stem}.json"
        if json_out.exists():
            data = json.loads(json_out.read_text())
            text = data.get("text", "").strip()
            transcript_path.write_text(text)
            json_out.unlink()  # clean up json, keep just .txt
            return text
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        print(f"\n  Error: {video_path.name}: {e}")
    return None


def transcribe_account(account: str):
    """Transcribe all videos for one account."""
    video_dir = VIDEOS_DIR / account
    if not video_dir.exists():
        print(f"No videos for @{account}")
        return

    out_dir = TRANSCRIPT_DIR / account
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    print(f"\n@{account}: {len(videos)} videos")

    # Skip already transcribed
    remaining = []
    for v in videos:
        txt = out_dir / f"{v.stem}.txt"
        if not txt.exists():
            remaining.append(v)

    print(f"  {len(videos) - len(remaining)} already transcribed, {len(remaining)} remaining")

    success = 0
    for video in tqdm(remaining, desc=f"@{account}"):
        txt_path = out_dir / f"{video.stem}.txt"
        text = transcribe_file(video, txt_path)
        if text is not None:
            success += 1

    print(f"  Transcribed: {success}/{len(remaining)}")


def merge_transcripts(account: str):
    """Merge transcripts into the posts CSV."""
    csv_path = DATA_DIR / f"{account}_posts.csv"
    trans_dir = TRANSCRIPT_DIR / account
    if not csv_path.exists() or not trans_dir.exists():
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    if "transcript" not in df.columns:
        df["transcript"] = ""

    matched = 0
    for idx, row in df.iterrows():
        vid = str(row["id"])
        txt_path = trans_dir / f"{vid}.txt"
        if txt_path.exists():
            df.at[idx, "transcript"] = txt_path.read_text()
            matched += 1

    df.to_csv(csv_path, index=False)
    print(f"  @{account}: {matched}/{len(df)} posts matched with transcripts")


def main():
    accounts = sys.argv[1:] if len(sys.argv) > 1 else ACCOUNTS

    for account in accounts:
        transcribe_account(account)
        merge_transcripts(account)


if __name__ == "__main__":
    main()
