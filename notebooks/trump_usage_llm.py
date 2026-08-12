"""
Judge whether TRUMP himself used each shared insult word as an insult AT A PERSON (vs random/topical
use, vs quoting others). Text-only, gemini-2.5-flash, pydantic-typed, cost-tracked, cached & resumable.

This same schema + prompt + call is embedded in insult_lineage_audit.ipynb; the notebook reads the
cache (trump_usage_llm.jsonl) so public re-runs never hit the API.

  python trump_usage_llm.py --limit 10     # COST TEST (do this first)
  python trump_usage_llm.py                 # full run over the remaining shared words (resumable)
"""
import argparse, csv, json, os, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal

HERE = Path(__file__).resolve().parent
ROOT = HERE
for cand in (HERE, *HERE.parents):                 # find the data dir wherever this lives
    if (cand / "data" / "analysis" / "curated_insults.csv").exists():
        ROOT = cand; break
for envp in (ROOT / ".env", ROOT.parent / ".env", HERE.parent / ".env"):
    if envp.exists(): load_dotenv(envp); break
AN = ROOT / "data" / "analysis"
TRUMP_DIR = ROOT / "data" / "x" / "trump_archive"
OUT = AN / "trump_usage_llm.jsonl"
MODEL = "gemini-2.5-flash"
RATE_IN, RATE_OUT = 0.30 / 1e6, 2.50 / 1e6
BILLING = ["billing", "quota", "credit", "resource_exhausted", "prepayment", "limit exceeded"]

class TrumpUsage(BaseModel):
    word: str
    verdict: Literal["insult_at_person", "random_or_topical", "quoting_others", "mixed"]
    used_as_insult: bool = Field(description="True if Trump himself weaponized this word at a person in at least one tweet")
    evidence_tweet: str = Field(description="the single tweet that best shows the verdict, verbatim (trimmed ok)")
    rationale: str = Field(description="one sentence, concrete")

# --- shared words = curated insults that also appear (lexically) in Trump's tweets ---
lex = list(csv.DictReader(open(AN / "curated_insults.csv")))
for r in lex:
    r["term"] = r["term"].strip().lower()
    r["is_phrase"] = (" " in r["term"]) or ("-" in r["term"])

tweets = []
for fn in ["trump_bf_office.csv", "trump_in_office.csv"]:
    for row in csv.DictReader(open(TRUMP_DIR / fn), skipinitialspace=True):
        t = row.get("Tweet Text") or ""
        if not t.startswith("RT @"): tweets.append(t)
low = [t.lower() for t in tweets]
tok = set()
for l in low: tok.update(re.findall(r"[a-z']+", l))

def samples(term, is_phrase, k=12):
    pat = re.compile(re.escape(term) if is_phrase else r"\b" + re.escape(term) + r"\b")
    hits = [t for t, l in zip(tweets, low) if pat.search(l)]
    # surface the insult-likely ones first (has @handle or a Capitalized name)
    hits.sort(key=lambda t: (0 if ("@" in t or re.search(r"\b[A-Z][a-z]+", t)) else 1, len(t)))
    return [t.replace("\n", " ")[:220] for t in hits[:k]]

def is_shared(term, is_phrase):
    return (term in " \n ".join(low)) if is_phrase else (term in tok)

shared = []
seen = set()
for r in lex:
    if r["term"] in seen: continue
    seen.add(r["term"])
    if is_shared(r["term"], r["is_phrase"]):
        shared.append(r)

PROMPT = """You are auditing whether Donald Trump himself used a word as an INSULT AIMED AT A PERSON on Twitter.

Word: "{word}"

Here are Trump's tweets containing it (authored, not retweets):
{tweets}

Question: In these tweets, did Trump use "{word}" as an insult directed at a person or group?
- insult_at_person: he calls someone this / uses it to demean a person or group (e.g. "Weirdo Tom Steyer", "Dumbass @X")
- random_or_topical: he uses it for non-insult things (food, money/"gross domestic", a place, a trade term, a URL, a proper noun)
- quoting_others: the insult in the tweet is someone ELSE's words that he is reporting/criticizing, not his own
- mixed: genuinely both across the tweets
Set used_as_insult=true only if at least one tweet is a real personal insult by Trump.
Return the single most representative evidence tweet and a one-sentence rationale."""

def run(limit, cost_cap):
    done = set()
    if OUT.exists():
        done = {json.loads(l)["word"] for l in open(OUT)}
    todo = [r for r in shared if r["term"] not in done]
    if limit: todo = todo[:limit]
    print(f"shared words total={len(shared)} | already judged={len(done)} | this run={len(todo)} | model={MODEL}")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    total = 0.0
    with open(OUT, "a") as f:
        for i, r in enumerate(todo, 1):
            ex = samples(r["term"], r["is_phrase"])
            prompt = PROMPT.format(word=r["term"], tweets="\n".join(f"- {t}" for t in ex))
            try:
                resp = client.models.generate_content(
                    model=MODEL, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_schema=TrumpUsage, temperature=0),
                )
                u = resp.usage_metadata
                cost = (u.prompt_token_count or 0) * RATE_IN + (u.candidates_token_count or 0) * RATE_OUT
                total += cost
                obj = json.loads(resp.text)
                obj["_n_tweets"] = len(ex)
                f.write(json.dumps(obj) + "\n"); f.flush()
                print(f"  [{i}/{len(todo)}] {r['term']:16} -> {obj['verdict']:20} used={obj['used_as_insult']!s:5} ${cost:.5f}  “{obj['evidence_tweet'][:55]}”")
            except Exception as e:
                if any(b in str(e).lower() for b in BILLING):
                    print(f"FATAL billing/quota on {r['term']}: {e}\n  STOP."); sys.exit(2)
                print(f"  ERR {r['term']}: {e}")
            if total >= cost_cap:
                print(f"cost cap ${cost_cap} hit"); break
            time.sleep(2)
    print(f"\nsession cost ${total:.4f}  (avg ${total/max(len(todo),1):.5f}/word) -> extrapolated full {len(shared)} = ${total/max(len(todo),1)*len(shared):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cost-limit", type=float, default=0.50)
    a = ap.parse_args()
    run(a.limit, a.cost_limit)
