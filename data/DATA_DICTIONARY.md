# DATA DICTIONARY — Stage 1 & Stage 2
*The shared reference for every field: what it means, what inputs produced it, and its known caveats.*

## The provenance chain

```
LAYER 0 (raw & deterministic)                LAYER 1 (Stage 1 - describe)         LAYER 2 (Stage 2 - classify)
video .mp4 ─────────────────────┐
metadata CSV (caption, date …) ─┤
Whisper transcript (cleaned) ───┤
easyocr overlay text ───────────┼──► gemini-2.5-pro (video-native) ──► description JSON ──► gemini-2.5-pro (text-only)
pol-face verified faces ────────┤         one call per post                 14 fields          one call per post
ACRCloud music timeline ────────┘                                                              function + treatments + craft
```
Every layer is on disk; any Stage-2 label traces back through its description to raw pixels/audio.

### Side-tool: the mechanism SWEEP (heuristic — NOT the Stage-2 classifier)
`mechanism_sweep.py` is a **third, read-only heuristic layer** on top of Stage 1. It applies Abidin-derived keyword/regex rules to the **Stage-1 descriptions + OCR + captions + music fingerprints** and writes candidate-mechanism counts to `data/eval/mechanism_sweep.json`. It was built as an **exploratory screen** to decide which attention mechanisms earned a slot in the classifier — *not* as a final measure. It is neither `describe.py` (it consumes describe's output) nor `classify_function.py` (it never calls an LLM).

**Two different instruments measure "mechanisms" — DO NOT mix their numbers in one claim:**

| | mechanism sweep (`mechanism_sweep.py`) | Stage-2 classifier (`classify_function.py`) |
|---|---|---|
| method | keyword/regex heuristic over Stage-1 text | LLM judgment (gemini-2.5-pro) |
| labels | `screenshot_attack, no_context, static_card, second_person, tags_opponent`(voldemorting)`, trending_sound, lip_sync…` | 12-vocab: `screenshot_receipts, no_context_bait, rage_bait_framing, distortion_mockery…` |
| output | `data/eval/mechanism_sweep.json` | `attention_mechanisms[]` in `function_*.jsonl` |
| status | exploratory / screening / second opinion | **AUTHORITATIVE for the thesis** (backs every cited mechanism number) |

The same concept reads very differently across the two — e.g. "no context" is `no_context` **12.8%** (sweep keyword, full corpus) vs `no_context_bait` **4.2%** (Stage-2 LLM, tighter definition). **Rule: cite the Stage-2 number for any mechanism it has a field for; use the sweep ONLY for signals the classifier lacks (e.g. voldemorting / does-the-attack-name-the-target).**

---

## LAYER 0 — raw & deterministic sources

| field / file | produced by | inputs | notes & caveats |
|---|---|---|---|
| `videos/{acct}/{id}.mp4` | `download.py` (yt-dlp, h264-forced) | TikTok CDN | Forced to h264 (`-f "b[vcodec*=264]/b"`) so audio isn't dropped on long videos. Carousels = `{id}_carousel.mp4` (slides 3s + audio via ffmpeg) |
| `data/{acct}_posts.csv` | `scrape.py` (yt-dlp metadata) | TikTok | 21 fields: `id, title`(=caption)`, upload_date`(UTC — runs ~5h ahead of ET)`, view/like/comment/repost_count, duration, track, url…` |
| `data/transcripts/{acct}/{id}.txt` | Whisper turbo (local) | .mp4 audio | Verbatim incl. bleeped forms (`f***`). Hallucination-cleaned (87 reviewed). Empty = music-only post. **Lyrics and speech are NOT distinguished** |
| `data/ocr/{acct}/{id}.json` | `ocr_overlays.py` (easyocr, frame-differencing) | .mp4 frames | `full_text, overlay_text[], n_frames, frame_times`. Text only — **emojis are NOT captured by OCR** (they come from Stage 1's multimodal read). Consistency vs Stage-1 ~0.65–0.74 |
| `../pol-face/output/presence_{acct}.csv` | pol-face (RetinaFace+ArcFace vs 383-person roster) | .mp4 frames | Per-scene person rows + confidence tiers. **Presence ≠ main character.** Roster-bounded; frontal-face; hand-adjudicated meme/composite supplement in `manual_appearances_{acct}.csv` |
| `data/music/{acct}/{id}.json` | `music_id.py` (ACRCloud fingerprint) | .mp4 audio windows | `distinct_tracks[], timeline[] (start, track, artist, score), silent_intro`. **Authoritative over any guess — human or model** |
| caption | `{acct}_posts.csv` `title` col; coding-sheet `caption` overrides where present | TikTok | The account's own words; where the emoji-bait lives |

---

## LAYER 1 — Stage-1 description (`data/descriptions/gemini_v2/{acct}/{id}.json`)

**Producer:** `describe_video.py` — gemini-2.5-pro, native video call. **Inputs per post:** the raw .mp4 itself + a context block containing caption, TikTok track name, cleaned transcript, OCR text, **authoritative face-IDs** (presence + manual CSVs), **authoritative music timeline** (ACRCloud), TODAY date-grounding. Validated: cast recall **0.956** on 604 held-out WH posts; 0 gross errors in a 24-post spot-check.

| field | type | definition | primarily derived from | caveats |
|---|---|---|---|---|
| `visual_action` | str | Neutral account of what happens on screen | video frames | |
| `cast[]` | list | Every political person/org in or centrally referenced by the post: `name, kind(person/org/unknown), how_identified(face/named_in_caption/named_in_audio/on_screen_text/voice/implied), role_in_post(speaking/spoken_about/both/shown_only), is_main(bool), evidence` | **face-ID CSVs (authoritative for faces)** + audio + OCR + caption + video | `is_main` = whom the post is ABOUT (may be unseen). Pop-culture cameos go to `cultural_references`, NOT cast. Face-ID = presence only; main-character is the model's judgment |
| `on_screen_text[]` | list | Overlay text items with role (caption/overlay/chyron/meme_text) | OCR (anchor) + video frames | Recall vs OCR ~0.63; OCR itself is the deterministic anchor |
| `spoken_summary` | str | What is said, reconciled with the transcript | transcript + audio | **A summary — paraphrases; swearing often cleaned away.** Use raw transcript for verbatim needs |
| `audio_music` | str | Music/SFX structure and its role/effect | ACRCloud timeline (authoritative for identity) + audio | Model may still name songs; **identity comes from the fingerprint only** |
| `editing_pacing` | str | Cuts, zooms, ramps, pacing | video | Free text — superseded for analysis by Stage-2 `production_style` |
| `emotions_signaled[]` | list | Emotions the post signals | whole-post read | **Post-level, not per-person** — per-person valence lives in Stage-2 `treatments` |
| `cultural_references[]` | list | Pop-culture borrowings incl. famous non-political people/characters | video + audio + caption | Known over-inclusion (artists' names, incidental logos) — treat as leads, not counts |
| `is_manipulated` | bool | **DEMOTED — do not use.** | — | Weak on world-knowledge judgments (F1 0.36 vs hand-coded tags). Never fed to Stage 2 |
| `manipulation_or_satire` | str | DESCRIPTIVE account of visible manipulation (composites, AI, warping) | video | Reliable for describe-what-you-see; unreliable for is-the-collateral-real |
| `emojis_present[]` | list | Emojis visible in the post | video (multimodal read) | OCR can't do emojis; this is the emoji source |
| `central_claim` | str | The post's core communicative claim | whole-post synthesis | |
| `intended_effect` | str | The reaction the post is engineered to produce | whole-post synthesis | |
| `uncertainty_flags[]` | list | Model's self-reported doubts | — | Honest grounding; drives review, not analysis |

---

## LAYER 2 — Stage-2 classification (`data/classification/function_{acct}.jsonl`)

**Producer:** `classify_function.py` — gemini-2.5-pro, text-only, temp 0.1, structured output.
**Inputs per post (the evidence payload):** Stage-1 fields `visual_action, cast, on_screen_text, spoken_summary, audio_music, editing_pacing, emotions_signaled, cultural_references, manipulation_or_satire, central_claim, intended_effect` **+ `caption`** (from posts CSV / coding sheet) **+ the account's SIDE_CONTEXT** (whose account it is; own vs opponent rosters) **+ the fixed prompt** (7 function categories, 8 context rules, the treatments scale incl. the domestic-partisan side rule).
**Deliberately excluded inputs:** `is_manipulated` (demoted boolean), `uncertainty_flags`, mechanism-sweep chips, engagement counts (no popularity contamination), the raw video.
**Reading note:** every Stage-2 field is judged in ONE call over the SAME full payload — the "decided from" column below names the fields that carry the DECISIVE signal for each output, not an input restriction.
**Rules inventory:** the 8 numbered context rules govern FUNCTION only. Other fields carry their rules in their schema definitions: treatments (all-pairs; exclude cameos; side = domestic partisan divide; non-substantive → register 0), is_peak (exactly one; framing tiebreaker — "X RIPS Y" packaging makes the mocked party peak), crudeness (own-expression only; lyrics can't raise it; **conservatism: torn → lower level**), has_dunk (witty/meme-work bar; slogans excluded), attention_mechanisms (12 fixed definitions; never guess trend status). Conservatism applies to the two numbers headed for copy: attack-share and crudeness — both are floors.

| field | type | definition | decided from | caveats |
|---|---|---|---|---|
| `function` | enum | PRIMARY function: `attack / promote_leader / promote_policy / patriotism / ceremonial / mobilize / other` — per the 8 in-prompt context rules (peak rule; attack needs a domestic-opponent target; policy needs an outcomes CLAIM; ceremonial is event-anchored; mobilize needs an explicit ask; **conservatism: never inflate attack**; non-substantive → other) | `central_claim` + `intended_effect` (what it's doing) · `cast` sides · caption/`on_screen_text` framing (chyrons like "RIPS" decide the margin) · `visual_action` | Attack-share is a **floor** by design |
| `treatments[]` | list | **Every subject-treatment pair** — one entry per person/collective the post meaningfully portrays: `subject, side(own/opponent/neutral), register(−3…+3), is_main_character, is_peak, evidence`. Dual-valence posts (glorify Biden −2 AND mock Trump +2) yield multiple entries | `cast` (who) · `visual_action`/`spoken_summary`/`on_screen_text` (how each is shown/spoken about) · `manipulation_or_satire` (distorted depictions) · `audio_music` (tonal framing) | Excludes background cameos/name-drops. `side` = DOMESTIC partisan divide: foreign adversaries, nonpartisan institutions, apolitical celebrities = `neutral`; celebrities with a partisan stance = `opponent` |
| `treatments[].register` | int | −3 hero-worship · −2 admiring · −1 warm/humanize · 0 neutral · +1 critical(substantive) · +2 mocking/ridicule · +3 hostile/taunting | the same treatment evidence, per subject | A bidirectional escalation scale. **Attack line = register ≥ +2 with side=opponent** |
| `treatments[].is_main_character` | bool | Whom the post is chiefly ABOUT (visible or not) | `central_claim` (who the post is ABOUT) · `cast.is_main` signals within the payload | Cross-checkable vs Stage-1 `cast.is_main` (QA agreement metric) |
| `treatments[].is_peak` | bool | The single strongest, most central treatment (exactly one per post) | where the post's energy builds: `visual_action` structure · `editing_pacing` · `intended_effect` · framing devices | Powers post-level charts; `peak_*` columns are derived from it |
| `peak_subject / peak_register / peak_side` | derived | Flattened copy of the is_peak treatment | derived in runner | Convenience columns only |
| `function_confidence` | enum | HIGH unambiguous / MEDIUM context-rules-needed / LOW torn | model self-report | LOW+MEDIUM = the validation triage set |
| `treatments_confidence` | enum | Confidence in the pair set | model self-report | |
| `function_rationale` | str | One sentence: the evidence that decided function, citing the rule applied | — | The audit trail |
| `production_style` | enum | `cinematic_produced / fast_montage / talking_head / static_graphic / raw_clip / other` | editing_pacing + visual_action + audio_music | LLM judgment. Powers the imitation-gap contrast |
| `has_dunk` | bool | Does the post land a compressed **dunk** — a WITTY, meme-work line (putdown, ironic twist, internet-native quip) built to be screenshotted. Earnest slogans/hype lines DON'T count | caption + on_screen_text + spoken_summary | Compression-as-dunk definition; bar tightened so "I am your voice"=False, "We're so back"=True |
| `peak_rationale` | str | One sentence: why the is_peak treatment is the peak vs the other pairs | same signals as `is_peak` — states the energy/structure argument explicitly | The audit trail |
| `crudeness` + `crudeness_evidence` | int 0–3 + str | Manner of the post, independent of profanity and target: 0 institutional · 1 edgy-informal · 2 crude (insults, appearance-mockery, innuendo) · 3 crass (vulgarity in own voice, body-shaming, scatological) | **the account's OWN expression**: caption · `on_screen_text` · `spoken_summary` (its speakers) · `visual_action` (its imagery) · `manipulation_or_satire` (gross-out/distorted depictions). **Soundtrack lyrics alone do NOT raise crudeness** (in-prompt rule) | Measures the "dark woke" register — "bad-built butch body" scores 2 with zero profanity; a rap track's language scores 0 |
| `attention_mechanisms[]` | list enum | ALL mechanisms used (12-item vocabulary: pop_culture_borrowing, meme_format, screenshot_receipts, no_context_bait, static_text_card, distortion_mockery, manipulation_fabrication, aura_farming, rage_bait_framing, gamification, call_and_response, news_jacking) | per mechanism: `cultural_references`+`audio_music` (pop-culture/meme) · `visual_action`+`on_screen_text` (receipts/cards/gamification) · caption (bait, call-and-response) · `manipulation_or_satire` (distortion/fabrication) · `editing_pacing` (aura edits) · `central_claim` (news-jacking) | **The comprehensive mechanism read (LLM)** — supersedes the regex sweep for analysis; the sweep is a QA cross-check on its detectable subset. trending-sound stays deterministic (fingerprint) — the LLM cannot know trend status |
| `dunk_line` | str | The exact line, verbatim | same | Empty when has_dunk=false |

---

## LAYER 2-ADJACENT — deterministic classification companions

| field / file | produced by | inputs | definition & caveats |
|---|---|---|---|
| `profanity_{acct}.jsonl`: `profanities[], strong_profanities[], sources{}, strong, n` | `profanity_scan.py` (lexicon + masked/acronym regexes) | **raw transcript + OCR full_text + caption** (verbatim sources ONLY — not Stage-1 summaries) | Exact words cited per source. STRONG tier (unambiguous swears incl. `f***`, FAFO, LFG) vs MILD (damn/hell/crap). **Validated vs the WH-600 `expletive` tag: recall 83%; residual = audio-only expletives → all rates are FLOORS.** |
| `dunk_lines.csv`: one row per put-down (`account, date, dunk_line, punch, target, source, views, video_id`) | `extract_multidunk.py` | Stage-1 descriptions + caption/OCR + ACRCloud music ID | Every caption-sized put-down pulled per post, with a played song's lyric told apart from the account's own line. Used in `insult_wall.ipynb`. Stage-2's `has_dunk`/`dunk_line` keep one dunk per post; this keeps them all |
| `mechanism_sweep.json`: per-post `mechanisms[]` | `mechanism_sweep.py` (regex detectors) | descriptions + OCR + captions + music fingerprints | **A QA cross-check** — analysis-grade mechanisms come from Stage-2 `attention_mechanisms[]` (LLM). The sweep validates the overlap subset (screenshot/no_context/static_card agreement) and remains the source for `trending_sound` (deterministic, fingerprint-based — the LLM can't know trend status) |
| ~~Target table~~ → **superseded by Stage-2 `treatments[]`** | — | — | The Target dimension (who the post is about, own/opponent, incl. main character) is now fully recorded in `treatments[]` — ALL subject-treatment pairs, not a single main-character lookup |
| Target QA cross-check *(planned)* | deterministic script | Stage-1 `cast.is_main` × pol-face roster party affiliations | **Validation only, not a data product**: agreement rate between treatments' `is_main_character` subjects and Stage-1 `cast.is_main`, and between treatments' `side` and the roster's party affiliation. Disagreements = the review set |

---

## THE TWO AGGRESSION METRICS
- **Function attack-share** = share of posts with `function=attack` — "posts that exist primarily to attack." Clean semantics, a FLOOR.
- **Treatment aggression rate** = share of posts containing **ANY opponent-side treatment ≥ +2** — **the headline "how much more aggressive" number**, immune to function bucketing (an attack on the way to a donate-ask still counts). Use THIS for the escalation story.

## Audit log — loopholes found and fixed
Eight loopholes found and fixed (human-reviewed): 

- (1) attack-vs-mobilize precedence — an attack-then-explicit-ask post is bucketed `mobilize`, so aggression is measured via treatments, not function
- (2) subject-name fragmentation → a canonical-names rule plus an analysis-time alias merge (`data/analysis/subject_aliases.json`, a non-destructive 37-entry variant→canonical map applied in the notebook; the raw JSONL is never mutated. It merges true splits only — the model over-formalizes to legal names, `Charles Schumer`→`Chuck Schumer`, `Douglas Emhoff`→`Doug Emhoff`, and varies punctuation/case, `Jd/J.D. Vance`→`JD Vance` — while deliberately NOT merging distinct family members sharing a surname. Cut distinct subject strings 1,472→1,435)
- (3) quoted opponent vulgarity can't raise crudeness
- (4) a dunk must be deployed by the account/own side
- (5) distortion_mockery (ridicule-warping) vs manipulation_fabrication (presented-as-real/seamless fiction) — the demoted thing was Stage-1's `is_manipulated` BOOLEAN (a world-knowledge judgment), not manipulation-as-mechanism (one of the 12 attention mechanisms)
- (6) confidence calibrated to an observable criterion (rule-decided → ≤MEDIUM), since MEDIUMs are the flip-prone class across runs
- (7) "America"/nation/military-institution/national teams pinned NEUTRAL, so own-side glorification stays party-only and apples-to-apples
- (8) the is_peak invariant is enforced in the runner (auto-fix to max |register|, logged in the `peak_autofixed` column).

Accepted limitations: no per-mechanism evidence strings; collective and individual pairs coexist (chart-level dedup awareness); own-side criticism posts → `other`, with the pair carrying the signal.

## EXPORT — `data/classification/stage2_classification.csv`
One row per post: identity (`account, video_id, upload_date, url` from posts CSV) + all Stage-2 fields (treatments serialized; peak flattened) + profanity columns joined by `video_id`. A long-format `stage2_treatments.csv` (one row per subject-treatment pair) accompanies it for per-person and network analysis.

## Standing rules embedded in this design
1. **Accuracy > deterministic/free** — judgments go to the LLM; lexical/lookup stays deterministic.
2. **No re-asking what Stage 1 already tags** — cast, emotions, mechanisms are consumed, not re-derived.
3. **Contamination rule** — Stage 2 sees evidence, never Stage-1 judgments (`is_manipulated`) or engagement data.
4. **Fingerprint > any guess** for music; **face-ID = presence, not main-character**.
5. **Every judgment carries its evidence** — rationales/evidence strings at every layer; any number in copy traces to specific posts.
