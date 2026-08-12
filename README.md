# *Dirty Dems* — code, data & methodology

This is the data and code behind a data journalism thesis on how the Democratic Party's official TikTok account, **@democrats**, changed its voice after the 2024 election — turning from policy and get-out-the-vote messaging toward insults, crudeness and spectacle — measured against the two flagship Republican accounts, the RNC's **@republicans** and the Trump administration's **@whitehouse**.

This analysis included every post those three accounts published from their inception through July 31, 2026. In full, **4,014** posts were analyzed: **2,881** by @democrats, **908** by @whitehouse and **225** by @republicans.

Generally, TikTok posts resist analysis at scale because their meaning is distributed across channels: spoken dialogue, captions, text burned into the video, music, editing, meme formats, faces and visual context. A transcript alone misses most of what a viewer sees, and a caption alone is often deliberately meaningless without the clip beneath it.

The innovation here is a solution where each of those videos was first bundled with machine-extracted signals, including a speech transcript, on-screen text, face identifications and the soundtrack's identity. Then, a multimodal large language model (Google's Gemini 2.5 Pro) turned those signals into a structured description: what happens, who appears, what is said and shown, and what is claimed. A second, text-only pass by the model then classified those descriptions into the measures used throughout: each post's primary purpose; how each political figure is portrayed, from glorification to attack; the crudeness of the account's manner; and techniques such as rage-bait framing and the put-downs the internet calls "dunks."

<img width="1288" height="801" alt="Image" src="https://github.com/user-attachments/assets/c7ea9155-430d-4fd8-acf4-de09714c68ed" />

This folder contains that pipeline, the data it derived, and the auditable notebooks behind every cited number. 

---

## What's in here

```
dirty-dems/
├── README.md                     ← this file: the folder map + the full methodology (postscript)
├── requirements.txt              ← Python deps for the analysis notebooks
│
├── signals2text/                 ── THE PIPELINE that produced the data (watch first, judge second)
│   ├── transcribe.py, transcribe_scope.py   speech transcript (Whisper)
│   ├── ocr_overlays.py                       on-screen text (EasyOCR + PySceneDetect)
│   ├── music_id.py                           soundtrack identity (ACRCloud)
│   ├── convert_carousels.py                  slideshow posts → video
│   ├── faceid/                               faces (InsightFace vs a YouGov roster) — core scripts
│   ├── describe_video.py  (+ description_config.py, description_schema.py)   Stage 1: describe
│   ├── classify_function.py                  Stage 2: classify
│   ├── profanity_scan.py, extract_multidunk.py   downstream extraction (profanity, dunk lines)
│   ├── classify_val_claude.py, eval_descriptions.py, build_validation_sample.py   validation
│   ├── mechanism_sweep.py                     exploratory keyword screen (NOT the classifier)
│   └── requirements.txt, .env.example
│
├── notebooks/                    ── THE AUDITABLE ANALYSIS (final, executed — every choice annotated inline)
│   ├── corpus_eda.ipynb              exploratory pass: corpus, field distributions, time, account/era splits, caveats
│   ├── insult_lineage_audit.ipynb    which insults are shared with Trump's own tweets vs the account's own — traced back to the vetted walls
│   ├── insult_wall.ipynb             extracted insult lines by account (voice / target / register) + the vetted Trump/Vance/profanity walls, each line linked to its post
│   └── trump_usage_llm.py                     one-off LLM pass (its verdicts are cached in data/, so notebooks never call the API)
│
└── data/                         ── EVERYTHING THE PIPELINE DERIVED (raw video is ringfenced out)
    ├── DATA_DICTIONARY.md                    definitions for every classified field the notebooks use
    ├── descriptions/                         Stage-1 descriptions — the plain-language, auditable layer
    │   ├── gemini_v2/{account}/              the authoritative run — all 4,014 posts, used everywhere downstream
    │   ├── gemini/  +  claude/               the same ~60 posts described by each model — the bake-off that chose Gemini
    │   └── claude_val/                       a second, independent Claude pass, kept for cross-checking the classifications
    ├── classification/                       Stage-2 labels (+ profanity, charge)
    ├── posts/                                post IDs + metadata (re-collect video from these)
    ├── analysis/                             derived artifacts (dunk lines, curated lexicons)
    ├── x/trump_archive/                      Trump's archived tweets (insult-lineage source)
    └── walls_print.md                        hand-locked "according to Dems" insult walls (a notebook input)
```

To rebuild the data from scratch, see **[Running the pipeline](#running-the-pipeline)** below.

---

#### Postscript

## Teaching a bot that "thank you, mr. president!" isn't always a thank you


Stripped all the way back, this project taught a machine something any person already knows: in politics, words often don't mean what they say. It came out of a slog. Last spring I [spent two weeks]([url](https://wongpeiting.github.io/peak-meme)) hand-classifying 600 @whitehouse TikToks for another story — watching every single one and tagging it by subject, how the meme was packaged, and how it was cut.

<img width="1800" height="1110" alt="Image" src="https://github.com/user-attachments/assets/981beb24-b3a5-428d-b1f0-f94f78b49f38" />

Two things were clear by the end. The judgments were rarely hard for a human: the sarcasm, the references, the difference between a tribute and a taunt are legible within seconds of watching. But every judgment needed the whole post — a caption meant nothing without the video under it, and the music routinely reversed what the text seemed to say. And watching does not scale: 600 posts took a fortnight, and this story needed 4,014.

The pipeline described below, which I call signals2text, is my answer to that: build a machine that can watch first and judge second.

<img width="1136" height="839" alt="Image" src="https://github.com/user-attachments/assets/27453ec7-8093-46b9-8876-41129deb0fb4" />

The 600 hand-coded posts served as the pipeline's source of truth. For 50 of these posts, I tested competing LLMs, which were scored against my hand-coding. In each turn, I looked to see whether the machine named the right people, read the on-screen text, caught what was faked, and stayed faithful to my overall interpretation. I kept refining the instructions until the descriptions closely matched my reading of those 50. Google's Gemini 2.5 Pro described them most faithfully. On the remaining 550 hand-coded posts — the ones I could check against my own labels — it identified the right people in roughly 96 percent, the foundation the rest of the analysis rests on. I then ran it across the full corpus of 4,014 posts.

<img width="1305" height="810" alt="Image" src="https://github.com/user-attachments/assets/fc41c1ce-615d-46a6-83ad-d1f96f650840" />

## Peculiarities of political sarcasm

Insults are easy to miscount, especially when sarcasm inverts a line's surface meaning. Read literally, "thank you, mr. president!" is gratitude. On the TikTok feed, it captions the @democrats' "fat chud rules" post. In other examples, "Bro is definitely not mad" asserts calm about a man melting down and "look how cute 🤗" decorates mockery. A keyword scan will read all three as neutral or warm as the insult lives not in the words but in the mismatch between the words and the video underneath them — which is why classification had to work from every single signal baked into one post.

<img width="1800" height="1120" alt="Image" src="https://github.com/user-attachments/assets/1b2219eb-7fa6-4a5e-8982-7c7be6c0c44d" />

These signals were first extracted in separate passes before the model sees the video, using the
following tools:

- A speech transcript from OpenAI's Whisper turbo model.
- On-screen text read by EasyOCR from one representative frame per scene, with scenes cut by
  PySceneDetect.
- Face identifications from InsightFace's buffalo_l model matched against a reference roster of U.S.
  political figures from YouGov.
- And the soundtrack's identity from ACRCloud, a music-recognition service.

There were quirks to fix. Whisper polluted the transcript by mishearing lyric-heavy audio, so I instructed
the describing model to listen to the audio itself and note when the two disagree. The model is also
structurally bad at telling real from fake — it flags events it doesn't recognize as fabrications — so I
told it to trust the video as real and flag only manipulation it could visibly see, then checked those
calls by hand.

With the signals in hand, the pipeline worked in two phases.

In the first, Gemini 2.5 Pro watches each video alongside the signal bundle and writes a structured
description of the post, including who appears, what happens on screen, what is said and sung, what the
post claims and what effect it is built to produce.

In the second, the same model — reading only that written description, never the video — classifies the
post: its primary function; how it treats each person it features, on a scale from worship (−3) to
hostility (+3); how crude its own voice is, from institutional (0) to crass (3); which of twelve
attention-getting mechanisms it uses (rage-bait framing, aura-farming, meme formats, and the like); and
whether it lands a "dunk," or an insult packaged as a one-line, quotable put-down.

Splitting watching from judging means every verdict is anchored to a written description that can be read,
checked against the video, and disputed line by line. Wherever a post was borderline, the classifier was
instructed to take the milder reading, so the attack and crudeness figures are floors, not ceilings.

The classifier's labels were also checked against a fully independent second model — Anthropic's Claude Sonnet 4.6 — which, on a sample deliberately weighted toward the posts the model itself was least sure about, agreed on 87 percent of attack calls (whether a post is an attack) and 91 percent at the crude-or-crass line (a crudeness of 2 or higher). Human checks were targeted rather than sampled: every insult and profanity line shown in the piece's visualizations was checked against its source video — about 400 posts in all — and roughly 88 percent of the lines the machine had pulled were confirmed; the rest were wrong or borrowed and were cut by hand. Dozens of the machine's descriptions were spot-checked field by field, with no errors in the core classifications (function, tone, target); the minor issues that did surface were confined to peripheral details, such as a misidentified song or an over-eager identification of cultural references.

## How the LLM figured insults

Counting attacks is only half the measurement; the other half is extracting the insults themselves. The
unit chosen for this was the "dunk", which I consider the platform's native unit of combat. A TikTok
attack usually lands in a caption-sized line built to be quoted, screenshotted and reposted, and speaks in
references you either catch or you don't (read: iykyk). The choice also mirrors the DNC's own war-room
philosophy as the party wanted "people in charge who know how to dunk."

The LLM is taught to judge a dunk by how a line is delivered, not what it argues, as a dry policy hit is
not a dunk but a cocky brag about your own side can be. So each post is asked three questions: Is it an
attack? Does that attack still carry an argument? Is it built as a performance, a line written to be
quoted?

Having the model pull the line itself — not just labeling the whole post an "attack" — is what makes the
insult wall visualizations possible. Each line is saved with who said it and who it targets, so individual
words can be counted, traced (to Trump's own tweets, among other places), and checked against the video.
Only the account's own words count toward its tally; a song it played or a line it quoted does not —
unless the post stamps that line on screen and builds the joke around it.

<img width="1800" height="2669" alt="Image" src="https://github.com/user-attachments/assets/a2ec861a-ae95-4d30-a120-1e36aedc40ef" />

## Why these accounts, and why TikTok

With the Democrats out of power, the party has no government feed to answer @whitehouse. Its senators and
governors run their own, largely regional, accounts, so the nearest thing to a national megaphone is the
Democratic National Committee's. I built the comparison around the DNC's flagship, @democrats, and set it
against the two biggest official accounts on the Republican side — @whitehouse, run by the Trump
administration, and @republicans, the RNC's. All three are staffed, tightly managed feeds — each its
side's official institutional voice.

I focused on TikTok because it is where I observed parties to be the least guarded. My earlier analysis of
the White House's first 600 TikTok posts found that, while it often cross-posts its TikTok content on
Instagram, it kept 24 of its 35 posts that carry profanity off Instagram. That reinforced my
expectation that if official politics were loosening its collar, TikTok is where it would happen first, and
most visibly. It is also the most meme-driven of the major feeds, where political operatives could fold
serious messaging into a joke.

---

## Running the pipeline

The `.mp4` files are ringfenced, but the layers the pipeline derived from them ship under `data/`, so the
analysis is reproducible from the descriptions down without re-fetching video. To rebuild from scratch,
re-collect the posts from the ID lists in `data/posts/*.csv`, then run the stages in order:

| Order | Stage | Script(s) | Reads | Writes |
|------:|-------|-----------|-------|--------|
| 1 | Slideshow → video | `convert_carousels.py` | image carousels | one `.mp4` per post |
| 2 | Speech transcript | `transcribe.py` | video | `transcripts/{account}/{id}.txt` |
| 2 | On-screen text | `ocr_overlays.py` | video (scene keyframes) | `ocr/{account}/{id}.json` |
| 2 | Music ID | `music_id.py` | video audio → ACRCloud | `music/{account}/{id}.json` |
| 2 | Faces | `faceid/` | video frames | `presence_{account}.csv` |
| 3 | **Describe (Stage 1)** | `describe_video.py` | video + all four signals | `descriptions/gemini_v2/{account}/{id}.json` |
| 4 | **Classify (Stage 2)** | `classify_function.py` | descriptions + posts | `classification/function_{account}.jsonl` |
| 4 | Profanity / dunks | `profanity_scan.py`, `extract_multidunk.py` | descriptions + signals | `classification/profanity_*`, dunk lines |
| 5 | Independent check | `classify_val_claude.py`, `eval_descriptions.py` | descriptions | second-model validation |

Stage 2 reads only the Stage-1 descriptions, which is why it Stage 2 can be re-run offline. `mechanism_sweep.py` is an exploratory keyword screen kept for provenance, and is not the classifier. The `faceid/` scripts are the core of a separate project (**pol-face**), which builds the reference roster and detects faces across the corpus.
