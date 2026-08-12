"""
Stage 2 — the ONE structured classification (advisor-approved scope):
FUNCTION + TREATMENTS (all subject-treatment pairs) from Stage-1 descriptions. TEXT-ONLY, cheap.

Reads description EVIDENCE only (never Stage-1 booleans / mechanism chips — contamination
rule). One prompt, context rules for margins. Safety rig cloned from describe_video.py:
single stream, MIN_SLEEP, backoff, circuit breaker, billing kill-switch, cost cap,
resumable (skips ids already in the output file), checkpointing.

  python classify_function.py --limit 10 --cost-limit 1.00       # test-10 first (doctrine)
  python classify_function.py --cost-limit 25.00                  # full corpus
Output: data/classification/function_{account}.jsonl (one line per post, resumable)
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("../.env")
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal

HERE = Path(__file__).parent
DESC = HERE / "data/descriptions/gemini_v2"
OUT = HERE / "data/classification"
OUT.mkdir(exist_ok=True)
ACCTS = ["whitehouse", "democrats", "republicans"]

MODEL = "gemini-2.5-pro"
MIN_SLEEP = 3.0
RATE_IN, RATE_OUT = 1.25 / 1e6, 10.0 / 1e6   # 2.5 Pro per-token USD
BILLING_KEYWORDS = ["billing", "quota", "credit", "resource_exhausted", "prepayment"]


class Treatment(BaseModel):
    subject: str = Field(description="the person or collective being portrayed (name from cast, e.g. 'Donald Trump'; or 'the Democratic Party', 'the media', 'America')")
    side: Literal["own", "opponent", "neutral"] = Field(description="the subject's side relative to the posting account")
    register: int = Field(ge=-3, le=3, description="-3 hero-worship .. -1 warm .. 0 neutral .. +1 critical .. +2 mocking .. +3 hostile/taunting")
    is_main_character: bool = Field(description="is this subject the post's MAIN character — whom the post is chiefly ABOUT (visible or not)")
    is_peak: bool = Field(description="is this the post's PEAK treatment — the strongest, most central portrayal. EXACTLY ONE treatment must have is_peak=true")
    evidence: str = Field(description="VERY SHORT phrase: what in the post treats them this way")


class FunctionLabel(BaseModel):
    function: Literal["attack", "promote_leader", "promote_policy",
                      "patriotism", "ceremonial", "mobilize", "other"] = Field(
        description="the post's PRIMARY function per the context rules")
    treatments: list[Treatment] = Field(
        description="EVERY subject the post meaningfully portrays with valence (or clear neutrality "
                    "as its topic) — one entry per subject-treatment pair. Include all ridiculed "
                    "targets AND all celebrated own-side figures. Do NOT include background cameos "
                    "or passing name-drops with no treatment. Typically 1-4 entries.")
    function_confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="HIGH = unambiguous; MEDIUM = context rules needed to resolve; LOW = genuinely torn between two functions")
    treatments_confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="confidence in the treatment pairs (subjects, sides and scores)")
    peak_rationale: str = Field(
        description="VERY SHORT (one sentence): why the is_peak treatment is the post's PEAK — "
                    "what the post builds to / spends its energy on, especially versus the other pairs")
    crudeness: int = Field(ge=0, le=3,
        description="how CRUDE/CRASS the post's manner is (independent of profanity and of whom it "
                    "targets): 0=institutional/polite; 1=edgy-informal (internet slang, snark, "
                    "cheeky memes); 2=crude (personal insults, schoolyard nicknames, gross-out or "
                    "mild sexual innuendo, mocking someone's appearance); 3=crass (explicit "
                    "vulgarity in the account's own voice, body-shaming, sexual/scatological "
                    "content, dehumanizing language). Judge the ACCOUNT'S OWN expression — caption, "
                    "overlay text, its speakers, its imagery. Soundtrack lyrics alone do NOT raise "
                    "crudeness. Vulgarity merely QUOTED/shown as evidence of an opponent's behavior does not "
                    "raise crudeness — only vulgarity deployed as the account's own rhetoric does. "
                    "CONSERVATISM: when torn between two levels, choose the LOWER — never inflate")
    crudeness_evidence: str = Field(
        description="VERY SHORT phrase citing what earns the crudeness score; empty string if 0")
    attention_mechanisms: list[Literal[
        "pop_culture_borrowing",   # meme/movie/song/celebrity culture used as the vehicle
        "meme_format",             # a recognizable meme template or trend format
        "screenshot_receipts",     # opponent's post/headline/clip shown as evidence to react to
        "no_context_bait",         # deliberately cryptic/emoji-only, forces decoding
        "static_text_card",        # shareable text-graphic built for screenshots
        "distortion_mockery",      # warped/exaggerated/deep-fried depiction of a person FOR ridicule
        "manipulation_fabrication",# AI/composited/doctored content presented AS IF real or as seamless fiction (not mere ridicule-warping)
        "aura_farming",            # cinematic glorification edit (dramatic score, slow-mo, hero framing)
        "rage_bait_framing",       # engineered to provoke angry engagement (taunt, gloat, intentional provocation)
        "gamification",            # game UI, scores, brackets, playable framing
        "call_and_response",       # direct address soliciting audience reaction/comments
        "news_jacking",            # riding a breaking news moment/controversy
    ]] = Field(
        description="ALL attention mechanisms the post uses (empty list if none). Judge from the "
                    "description evidence; do not guess trending-sound status (measured separately)")
    function_rationale: str = Field(
        description="VERY SHORT (one sentence): the description evidence that decided the FUNCTION, citing which context rule applied if any")
    production_style: Literal["cinematic_produced", "fast_montage", "talking_head",
                              "static_graphic", "raw_clip", "other"] = Field(
        description="the post's dominant production form: cinematic_produced=polished hero-edit "
                    "(dramatic score, graded, produced); fast_montage=rapid-cut compilation; "
                    "talking_head=person speaking to camera / podium clip; static_graphic=text card "
                    "or still image(s); raw_clip=unedited/lightly-edited found or event footage")
    has_dunk: bool = Field(
        description="does the post deliver a compressed 'dunk' — a short, WITTY line (in caption, "
                    "overlay, or speech) doing meme-work: a putdown, ironic twist, or "
                    "internet-native quip engineered to be screenshotted/memeified. The line must be "
                    "DEPLOYED BY the account or its own side (caption, overlay, its speakers, its "
                    "chosen meme-audio) — a line spoken by the TARGET inside mocked footage is not "
                    "the account's dunk. Earnest slogans, hype lines, and ordinary quotes do NOT "
                    "count — the line must carry wit, snark, or cultural-reference play")
    dunk_line: str = Field(
        description="if has_dunk: the exact line, quoted verbatim from caption/overlay/speech; else empty string")


SIDE_CONTEXT = {
    "whitehouse": "This is the OFFICIAL WHITE HOUSE account under President Donald Trump "
                  "(Republican administration, Jan 2025–present). OWN side = Donald Trump, his "
                  "family, cabinet (Vance, Leavitt, Hegseth, Rubio...), Republicans, ICE/military "
                  "under his command. OPPONENT side = Democrats, Biden, Harris, liberal critics, "
                  "protesters against Trump, and media outlets framed as hostile. A menacing or "
                  "aggressive portrayal of TRUMP on this account is glorification (aura), not attack.",
    "democrats": "This is the OFFICIAL DNC account. OWN side = Democratic figures (Biden, Harris, "
                 "Obama, congressional Democrats, Democratic governors/candidates). OPPONENT side = "
                 "Donald Trump, his family/cabinet, Republicans, MAGA figures, Musk and Trump-allied "
                 "media. This holds for the entire timespan (2022–present).",
    "republicans": "This is the OFFICIAL RNC account. OWN side = Donald Trump, Republicans, "
                   "conservative figures. OPPONENT side = Democrats, Biden, Harris, progressive "
                   "celebrities and activists.",
}

PROMPT = """You are classifying one TikTok post from an official US political account
({account}). ACCOUNT CONTEXT: {side_context}

Below is a rich, evidence-bearing DESCRIPTION of the post (produced from the
raw video, transcript, on-screen text, verified faces and fingerprinted music). Classify
the post's PRIMARY FUNCTION — what it is chiefly trying to DO — and its TREATMENTS
(every subject the post portrays, each with its own register).

FUNCTION (pick exactly one):
- attack          : discredit/mock/blame/alarm about the OPPONENT side (people or party)
- promote_leader  : build the image/persona of own-side figure(s) — strength, coolness, compassion
- promote_policy  : tout specific policies, achievements, legislation, economic results
- patriotism      : national pride, military, flag, American greatness NOT tied to a policy claim
- ceremonial      : holidays, sports champions, cultural events, observances
- mobilize        : explicit calls to vote, donate, join, turn out
- other           : none of the above fits (rare)

CONTEXT RULES for margins (apply in order):
1. If the post BOTH glorifies own side AND attacks the opponent, code the PEAK: whichever
   the post builds to / spends its energy on. If genuine mockery/hostility toward the
   opponent is present and central, attack wins. FRAMING DECIDES the margin: if the
   account's own packaging (caption, chyron, on-screen title like "X RIPS Y" / "roasts" /
   "destroys") presents the post as a confrontation, the attack IS central — code attack,
   and the mocked party is the peak treatment.
2. attack requires an opponent-side target (person, party, or their policy). Criticizing
   e.g. a foreign adversary while praising own leadership = promote_leader or patriotism.
3. patriotism vs promote_leader: if a specific own-side figure is the vehicle ("Trump
   embodies the flag"), code promote_leader. Anonymous flags/military/anthem = patriotism.
4. promote_policy vs promote_leader: policy needs a CLAIM about outcomes/plans (jobs,
   prices, border numbers, healthcare). Pure persona/aesthetic = promote_leader.
5. ceremonial vs patriotism: ceremonial is EVENT-anchored (Easter Egg Roll, championship
   visit, St. Patrick's). July-4-style flag content with no event = patriotism.
6. mobilize only when the ask is explicit (vote/donate/sign up), not implied enthusiasm.
7. CONSERVATISM: when torn between attack and any other function, code attack ONLY if
   mockery/hostility toward an opponent-side target is unambiguous and central — otherwise
   prefer the non-attack reading and set function_confidence MEDIUM or LOW. Never inflate attack.
8. Non-substantive posts (pure logistics, announcements without a claim, feed housekeeping)
   = other, register 0. A caption or hashtag alone cannot make a post an attack if the video
   itself is ceremonial/neutral.
9. attack vs mobilize: if the post CULMINATES in an explicit ask (vote/donate/sign up),
   code mobilize even when attack material precedes it — the ask is what the post is FOR.
   Attack material without a closing ask stays attack.
10. CONFIDENCE DISCIPLINE: if ANY numbered rule was needed to resolve the call (esp. the
   rule-1 framing tiebreaker or rule-7 conservatism), function_confidence must be at most
   MEDIUM. HIGH is reserved for posts needing no margin rule at all.

TREATMENTS — capture EVERY subject-treatment pair, not just one:
-3 hero-worship | -2 admiring/proud | -1 warm/humanize | 0 neutral/straight
+1 critical (substantive) | +2 mocking/ridicule | +3 hostile/taunting
List one entry per subject the post meaningfully portrays: every ridiculed target AND
every celebrated own-side figure gets its own pair (subject, side, register, evidence).
SUBJECT NAMES must be CANONICAL: the person's plain full name exactly as given in the cast
("Donald Trump", never "President Trump" / "The President (Donald Trump)" / honorifics);
for collectives use stable forms: "the Trump administration", "the Democratic Party",
"the Republican Party", "the media", "America". Never invent name variants.
A post that glorifies Biden (-2) while mocking Trump (+2) yields TWO entries. Exclude
background cameos and passing name-drops that carry no treatment. Typically 1-4 pairs.
Flags: mark is_main_character=true on the subject(s) the post is chiefly ABOUT (whether
or not they appear on screen). Mark is_peak=true on EXACTLY ONE entry — the strongest,
most central treatment (the post's peak; ties break toward the treatment the post builds to).
SIDE means the DOMESTIC partisan divide: opponent = the rival US party and its aligned
figures/media. Foreign adversaries (Putin, ISIS, cartels), nonpartisan institutions, and
apolitical celebrities = neutral. "America"/the nation itself, the military as an
institution, and national teams = ALWAYS neutral (both parties claim them). A celebrity who has taken a partisan stance against the
posting account's side (e.g. attacking its policies) counts as opponent.

DESCRIPTION:
{description}
"""


import csv as _csv

def load_captions():
    """caption per (account, video_id): posts CSV first (full corpus), coding sheet overrides."""
    caps = {}
    for a in ACCTS:
        p = HERE / f"data/{a}_posts.csv"
        if p.exists():
            for r in _csv.DictReader(open(p)):
                caps[(a, str(r["id"]))] = r.get("title") or r.get("caption") or ""
        cs = HERE / f"{a}_coding_sheet.csv"
        if cs.exists():
            for r in _csv.DictReader(open(cs)):
                if (r.get("caption") or "").strip():
                    caps[(a, r["video_id"])] = r["caption"]
    return caps


def desc_text(d, caption=""):
    # evidence fields only — strip Stage-1 judgments we don't want echoed (is_manipulated etc.)
    keep = ["visual_action", "cast", "on_screen_text", "spoken_summary", "audio_music",
            "editing_pacing", "emotions_signaled", "cultural_references",
            "manipulation_or_satire", "central_claim", "intended_effect"]
    slim = {k: d.get(k) for k in keep if d.get(k)}
    if caption:
        slim["caption"] = caption          # the account's own words on the post
    return json.dumps(slim, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", nargs="*", default=ACCTS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cost-limit", type=float, default=25.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    captions = load_captions()
    total_cost, consec_err, n_done = 0.0, 0, 0
    t0 = time.strftime("%H:%M:%S")
    print(f"[{t0}] model={MODEL} accounts={args.accounts} cost_limit=${args.cost_limit}")

    for a in args.accounts:
        outp = OUT / f"function_{a}.jsonl"
        done = set()
        if outp.exists():
            for line in open(outp):
                try: done.add(json.loads(line)["video_id"])
                except Exception: pass
        files = sorted(glob.glob(str(DESC / a / "*.json")))
        todo = [f for f in files if os.path.basename(f)[:-5] not in done]
        print(f"@{a}: {len(files)} descriptions · {len(done)} already classified · {len(todo)} to do")
        if args.dry_run:
            continue
        fout = open(outp, "a")
        for f in todo:
            if args.limit and n_done >= args.limit:
                print(f"limit {args.limit} reached"); fout.close(); report(n_done, total_cost); return
            if total_cost >= args.cost_limit:
                print(f"COST LIMIT ${args.cost_limit} reached — stopping"); fout.close(); report(n_done, total_cost); return
            vid = os.path.basename(f)[:-5]
            d = json.load(open(f))
            prompt = PROMPT.format(account=a, side_context=SIDE_CONTEXT[a],
                                   description=desc_text(d, captions.get((a, vid), "")))
            t = time.time()
            try:
                resp = client.models.generate_content(
                    model=MODEL, contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=FunctionLabel))
                lab = resp.parsed
                u = resp.usage_metadata
                cost = (u.prompt_token_count or 0) * RATE_IN + (u.candidates_token_count or 0) * RATE_OUT
                total_cost += cost
                # ENFORCE peak invariants (schema can't): exactly one is_peak
                peaks = [t for t in lab.treatments if t.is_peak]
                peak_fixed = ""
                if lab.treatments and len(peaks) != 1:
                    # auto-fix: strongest |register| wins (ties -> first); log for audit
                    best = max(lab.treatments, key=lambda t: abs(t.register))
                    for t in lab.treatments:
                        t.is_peak = (t is best)
                    peak_fixed = f"had {len(peaks)} peaks"
                    print(f"    ⚠ peak-invariant auto-fixed ({peak_fixed}) on {vid}")
                if not lab.treatments:
                    print(f"    ⚠ EMPTY treatments on {vid}")
                peak = next((t for t in lab.treatments if t.is_peak), None)
                row = {"account": a, "video_id": vid, "function": lab.function,
                       "treatments": [t.model_dump() for t in lab.treatments],
                       "peak_subject": peak.subject if peak else "",
                       "peak_register": peak.register if peak else 0,
                       "peak_side": peak.side if peak else "",
                       "peak_rationale": lab.peak_rationale,
                       "peak_autofixed": peak_fixed,
                       "crudeness": lab.crudeness,
                       "crudeness_evidence": lab.crudeness_evidence,
                       "attention_mechanisms": lab.attention_mechanisms,
                       "function_confidence": lab.function_confidence,
                       "treatments_confidence": lab.treatments_confidence,
                       "function_rationale": lab.function_rationale,
                       "production_style": lab.production_style,
                       "has_dunk": lab.has_dunk, "dunk_line": lab.dunk_line}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
                n_done += 1; consec_err = 0
                pairs = " · ".join(f"{t.subject[:18]}{'★' if t.is_peak else ''}{'Ⓜ' if t.is_main_character else ''}:{t.register:+d}({t.side[:3]})" for t in lab.treatments)
                print(f"[{time.strftime('%H:%M:%S')}] ✓ {vid} {lab.function:>14} [{pairs}] ${cost:.4f} ({time.time()-t:.1f}s) total=${total_cost:.4f}")
            except Exception as e:
                msg = str(e).lower()
                if any(k in msg for k in BILLING_KEYWORDS):
                    print(f"FATAL (billing/quota) on {vid}: {e}\n  STOP — check billing."); fout.close(); report(n_done, total_cost); sys.exit(2)
                consec_err += 1
                print(f"  ✗ {vid}: {str(e)[:160]} (consec {consec_err})")
                if consec_err >= 5:
                    print("CIRCUIT BREAKER: 5 consecutive errors — stopping."); fout.close(); report(n_done, total_cost); sys.exit(3)
                time.sleep(min(60, 2 ** consec_err * 3))
            el = time.time() - t
            if el < MIN_SLEEP:
                time.sleep(MIN_SLEEP - el)
        fout.close()
    report(n_done, total_cost)


def report(n, cost):
    print(f"[{time.strftime('%H:%M:%S')}] done: {n} classified · session cost ${cost:.4f}")


if __name__ == "__main__":
    main()
