"""
Convert carousel posts (slide images + audio) into .mp4 video files.

Finds audio-only downloads (.m4a/.mp3) that have slide images ({id}_01.jpg, etc)
but no .mp4, then stitches images into a video at 3s/slide with the audio track.

Works across all three pipelines:
  - US party accounts (videos/{account}/)
  - US politician accounts (videos/politicians/{account}/)
  - Singapore accounts (videos/singapore/{account}/)

Usage:
  python convert_carousels.py                        # All accounts, all pipelines
  python convert_carousels.py --party-only            # US party accounts only
  python convert_carousels.py --politicians-only      # US politicians only
  python convert_carousels.py --sg-only               # Singapore only
  python convert_carousels.py --accounts democrats papsingapore  # Specific accounts
  python convert_carousels.py --dry-run               # Show what would be converted
"""
import argparse, glob, json, os, subprocess, sys
from pathlib import Path

SLIDE_DURATION = 3  # seconds per slide


def find_carousels(vid_dir):
    """Find audio-only posts with slide images but no .mp4."""
    vid_dir = Path(vid_dir)
    if not vid_dir.exists():
        return []

    audio_ids = {}
    for f in vid_dir.iterdir():
        if f.suffix in ('.m4a', '.mp3'):
            audio_ids[f.stem] = f

    results = []
    for vid_id, audio_path in sorted(audio_ids.items()):
        mp4_path = vid_dir / f"{vid_id}.mp4"
        carousel_mp4 = vid_dir / f"{vid_id}_carousel.mp4"
        if mp4_path.exists() or carousel_mp4.exists():
            continue

        slides = sorted(vid_dir.glob(f"{vid_id}_*.jpg")) + sorted(vid_dir.glob(f"{vid_id}_*.jpeg"))
        # Filter out non-slide files (e.g. {id}_carousel.mp4 wouldn't match but be safe)
        slides = [s for s in slides if s.stem.startswith(vid_id + "_")]

        results.append({
            'id': vid_id,
            'audio': audio_path,
            'slides': slides,
            'has_slides': len(slides) > 0,
        })

    return results


def stitch_carousel(vid_id, slides, audio_path, output_path):
    """Convert slide images + audio into a single .mp4 video."""
    if not slides:
        # Audio-only, no slides — create video from audio with black screen
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=10",
            "-i", str(audio_path),
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-pix_fmt", "yuv420p",
            str(output_path),
        ]
    else:
        # Build concat file for ffmpeg
        concat_path = f"/tmp/concat_{vid_id}.txt"
        with open(concat_path, "w") as f:
            for slide in slides:
                f.write(f"file '{slide.resolve()}'\n")
                f.write(f"duration {SLIDE_DURATION}\n")
            # Repeat last frame to avoid ffmpeg cutting it short
            f.write(f"file '{slides[-1].resolve()}'\n")

        total_duration = len(slides) * SLIDE_DURATION

        if audio_path.exists():
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_path,
                "-i", str(audio_path),
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-t", str(total_duration),
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_path,
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                "-t", str(total_duration),
                str(output_path),
            ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
        else:
            stderr = result.stderr.decode('utf-8', errors='replace')[-200:]
            print(f"    ffmpeg error: {stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    ffmpeg timeout")
        return False
    finally:
        concat_file = Path(f"/tmp/concat_{vid_id}.txt")
        if concat_file.exists():
            concat_file.unlink()


def process_directory(vid_dir, label, dry_run=False):
    """Process all carousels in a single video directory."""
    carousels = find_carousels(vid_dir)
    if not carousels:
        return 0, 0, 0

    with_slides = [c for c in carousels if c['has_slides']]
    without_slides = [c for c in carousels if not c['has_slides']]

    print(f"  @{label}: {len(carousels)} carousels ({len(with_slides)} with slides, {len(without_slides)} without)")

    if dry_run:
        for c in without_slides:
            print(f"    {c['id']} — no slides (needs download_carousels.py first)")
        return 0, len(without_slides), 0

    converted = 0
    for c in with_slides:
        output = Path(vid_dir) / f"{c['id']}_carousel.mp4"
        ok = stitch_carousel(c['id'], c['slides'], c['audio'], output)
        if ok:
            converted += 1
            size_kb = output.stat().st_size / 1024
            print(f"    {c['id']} — {len(c['slides'])} slides → {size_kb:.0f}KB")
        else:
            print(f"    {c['id']} — FAILED")

    if without_slides:
        print(f"    {len(without_slides)} carousels need slide images first (run download_carousels.py)")

    return converted, len(without_slides), len(with_slides) - converted


def main():
    parser = argparse.ArgumentParser(description="Convert carousel slide images to .mp4 video")
    parser.add_argument("--party-only", action="store_true")
    parser.add_argument("--politicians-only", action="store_true")
    parser.add_argument("--sg-only", action="store_true")
    parser.add_argument("--accounts", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_party = not args.politicians_only and not args.sg_only
    run_sg = not args.party_only and not args.politicians_only
    run_pol = not args.party_only and not args.sg_only

    if args.sg_only:
        run_party, run_sg, run_pol = False, True, False
    if args.politicians_only:
        run_party, run_sg, run_pol = False, False, True
    if args.party_only:
        run_party, run_sg, run_pol = True, False, False

    total_converted = 0
    total_need_slides = 0
    total_failed = 0

    # US Party accounts
    if run_party:
        print("── US Party accounts ──")
        party = ["democrats", "republicans", "whitehouse", "senatedems", "senatedemshq", "dccc"]
        if args.accounts:
            party = [a for a in args.accounts if a in party]
        for acc in party:
            c, ns, f = process_directory(f"videos/{acc}", acc, args.dry_run)
            total_converted += c; total_need_slides += ns; total_failed += f

    # Singapore accounts
    if run_sg:
        print("── Singapore accounts ──")
        sg_dirs = sorted(Path("videos/singapore").iterdir()) if Path("videos/singapore").exists() else []
        sg_accs = [d.name for d in sg_dirs if d.is_dir()]
        if args.accounts:
            sg_accs = [a for a in args.accounts if a in sg_accs]
        for acc in sg_accs:
            c, ns, f = process_directory(f"videos/singapore/{acc}", acc, args.dry_run)
            total_converted += c; total_need_slides += ns; total_failed += f

    # US Politician accounts
    if run_pol:
        print("── US Politician accounts ──")
        pol_dirs = sorted(Path("videos/politicians").iterdir()) if Path("videos/politicians").exists() else []
        pol_accs = [d.name for d in pol_dirs if d.is_dir()]
        if args.accounts:
            all_special = set(["democrats", "republicans", "whitehouse", "senatedems", "senatedemshq", "dccc"])
            pol_accs = [a for a in args.accounts if a not in all_special]
        for acc in pol_accs:
            c, ns, f = process_directory(f"videos/politicians/{acc}", acc, args.dry_run)
            total_converted += c; total_need_slides += ns; total_failed += f

    print(f"\n{'='*60}")
    print(f"DONE: {total_converted} converted, {total_failed} failed, {total_need_slides} need slide images first")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
