"""Settings for the description layer (Stage 1). Mirrors config.py conventions."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- models ---
# CHOSEN ARM (2026-07-01): gemini-2.5-pro native video. Won/tied every task in the
# gold A/B (cast 0.82 vs 0.67, on-screen-text 0.54 vs 0.47), and it wrote the best descriptions.
# claude kept only for comparison; the rollout uses gemini.
GEMINI_MODEL = "gemini-2.5-pro"          # Arm A: native video — CHOSEN
CLAUDE_MODEL = "claude-sonnet-4-6"       # Arm B: frames + transcript/OCR (comparison only)

# --- paths ---
PILOT_MANIFEST = HERE / "data" / "pilot" / "pilot_manifest.json"
OCR_DIR = HERE / "data" / "ocr"
MUSIC_DIR = HERE / "data" / "music"                # {account}/{id}.json (ACRCloud timeline)
DESC_DIR = HERE / "data" / "descriptions"          # {engine}/{account}/{id}.json
VIDEOS = HERE / "videos"
# pol-face sits beside all-of-pol-tiktok: python-work/pol-face/output/presence_{account}.csv
FACE_DIR = HERE.parent.parent / "pol-face" / "output"

# --- Arm B frame sampling (Claude can't take raw video) ---
CLAUDE_MAX_FRAMES = 8                     # scene-sampled keyframes per video

# --- safety (cloned doctrine from embed_safe.py / POSTMORTEM_BILLING.md) ---
MIN_SLEEP = 3.0
MAX_SLEEP = 4.0
MAX_CONSECUTIVE_ERRORS = 5
INITIAL_BACKOFF = 5
MAX_BACKOFF = 120
COST_LIMIT_DEFAULT = 5.0                  # hard per-run ceiling for the pilot
FATAL_KEYWORDS = ["EXHAUSTED", "depleted", "billing", "spend cap", "prepayment", "quota"]

# Gemini video size handling (same thresholds as embed_safe.py)
INLINE_MAX = 20 * 1024 * 1024
FILES_API_MAX = 100 * 1024 * 1024

# Approximate pricing ($ per 1M tokens) — ESTIMATES for the cost report; refine with
# current published rates. Token COUNTS in the report are exact (from usage metadata).
RATES = {
    "gemini": {"in": 1.25, "out": 10.0},   # gemini-2.5-pro (estimate)
    "claude": {"in": 3.0, "out": 15.0},    # claude-sonnet-4-6 (estimate)
}
