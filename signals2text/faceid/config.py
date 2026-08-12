"""Single source of all tunable constants and paths for the face-ID pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PARTY = PROJECT_ROOT.parent / "all-of-pol-tiktok" / "party_accounts"
VIDEOS_ROOT = PARTY / "videos"
POSTS_DIR = PARTY / "data"
WATCHLIST_DIR = PROJECT_ROOT / "watchlist"
OUTPUT_DIR = PROJECT_ROOT / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"

# In-scope accounts for THIS piece (folder name under videos/ and data/<acct>_posts.csv).
ACCOUNTS = ["whitehouse", "republicans", "democrats"]

# --- Stage 2: adaptive sampling ---
MIN_READS = 5          # floor of reads for a normal-length scene
MIN_FPS = 3.0          # sampling density (frames/sec) for normal scenes
SHORT_SCENE_S = 1.5    # scenes shorter than this are sampled every frame

# --- Stage 1: scene detection ---
SCENE_THRESHOLD = 27.0  # PySceneDetect ContentDetector sensitivity (lower = more cuts)

# --- Stage 3: recognition (assist only in Plan 1) ---
MATCH_THRESHOLD = 0.50  # cosine; CALIBRATE before trusting. Plan-1 default guess.
POSSIBLY_BAND = 0.10    # band below MATCH_THRESHOLD that counts as "possibly"
MIN_FACE_PX = 24        # ignore faces whose bbox height < this many pixels

# --- depiction hints ---
PIP_MAX_AREA = 0.04     # face bbox area / frame area below this → "pip" hint
BLUR_MAX = 80.0         # Laplacian-variance below this → "graphic" hint

# --- presence (used in Plan 2; defined here so config is one place) ---
MIN_READS_PRESENT = 1

# --- reporting cuts (used in Plan 3) ---
CONFIDENCE_TIER_CUTS = {"high": 0.60, "medium": 0.45}  # max_similarity bands
# PROMINENCE_CUTS: compared against face_area_pct which is a 0–1 FRACTION (not a percentage).
# e.g. 0.10 means the face bbox covers ≥10% of the frame area.
PROMINENCE_CUTS = [0.10]  # face_area_pct thresholds for "platformed" reporting

# --- Plan 2: clustering + naming ---
MIN_CLUSTER_SIZE = 8         # a face must recur in >= this many frames to be a cluster
CLUSTER_MIN_SAMPLES = 4      # HDBSCAN conservativeness (higher = more noise)
CLUSTER_SELECTION_EPSILON = 0.0
MERGE_BY_CENTROID = False    # appearance-merge (Stage 4b) fuses similar-looking DIFFERENT people
                             # (e.g. Hegseth+Vance). With a reference whitelist we merge by IDENTITY
                             # instead (clusters that match the same politician), which is safe.
CLUSTER_MERGE_THRESHOLD = 0.40  # only used if MERGE_BY_CENTROID is True (no-watchlist mode)
AUTONAME_THRESHOLD = 0.45    # centroid<->reference cosine to auto-name a cluster
PASS2_THRESHOLD = 0.50       # pass-2 recall: assign gate-dropped clusters to nearest CONFIRMED identity
# --- scalable match-based identity (match_faces.py) — replaces clustering at 100k+ faces ---
MATCH_ANCHOR_THRESHOLD = 0.50   # per-face cosine to anchor an identity (lowered from 0.55: recall probe showed 0.55 missed real figures - Harris, Warren, Cruz...)
MATCH_RECOVER_THRESHOLD = 0.50  # per-face cosine to an in-domain centroid to recover a weak face
MIN_PERSON_FACES = 3            # keep brief but real appearances (Obama/Harris show up in just a few frames)
WHITELIST_GATE = True        # when a watchlist exists, DROP clusters that match no roster
                             # politician (the crowd/meme/celebrity noise). Analyst can rescue
                             # a wrongly-dropped cluster by naming it in the label sheet/app.
COHESION_TIER_CUTS = {"high": 0.55, "medium": 0.40}  # face<->centroid cosine bands
