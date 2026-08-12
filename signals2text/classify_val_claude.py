"""
Cross-model validation classifier (Arm B, stage 2): Claude reads the CLAUDE-arm description
of each validation post and answers the same five questions the human GUI asks. Fully
separate chain from the Gemini pipeline (its own descriptions, its own classifier), so
Gemini-vs-Claude agreement is a genuine cross-model reliability check.
  python classify_val_claude.py            # all claude_val descriptions not yet labeled
Writes data/analysis/xmodel_claude_labels.jsonl (resumable), prints per-post + total cost.
"""
import csv, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "data/analysis/xmodel_claude_labels.jsonl"
DESC = HERE / "data/descriptions/claude_val"
MODEL = "claude-sonnet-4-6"
PRICE_IN, PRICE_OUT = 3.0, 15.0     # $/M tokens, sonnet-class

import anthropic
from dotenv import load_dotenv
load_dotenv(HERE.parent / ".env")
client = anthropic.Anthropic()

caps = {}
for a in ("democrats", "whitehouse", "republicans"):
    for r in csv.DictReader(open(HERE / f"data/{a}_posts.csv")):
        caps[str(r["id"])] = (r.get("title") or "").strip()

PROMPT = """You are classifying one TikTok post by an official US political account, using a
written description of the video plus its caption. Answer conservatively: when a case is
borderline, choose the milder / "no" reading.

DESCRIPTION:
{desc}

CAPTION: {cap}

Answer these five questions about the post:
1. attack — Is the post's MAIN job to go after an opponent (discredit, mock, blame, alarm)?
   A post that mainly promotes its own side is not an attack even if it jabs in passing. y/n
2. trump_present — Does Donald Trump appear (footage, photo, voice, or named on screen)? y/n
3. trump_attacked — If present, is he treated with mockery or hostility? y/n (n if absent)
4. crudeness — The ACCOUNT'S OWN manner only, 0-3: 0 institutional · 1 edgy-informal ·
   2 crude (insults, appearance-mockery, innuendo) · 3 crass (vulgarity in its own voice,
   body-shaming). Soundtrack lyrics alone never raise the score; quoted opponents' words
   don't count as the account's own manner.
5. namecalling_only — Is the attack pure name-calling WITHOUT any accusation or policy
   point anywhere in the post? y/n (n if not an attack)

Reply with ONLY a JSON object: {{"attack":"y|n","trump_present":"y|n","trump_attacked":"y|n","crudeness":0-3,"namecalling_only":"y|n"}}"""

done = set()
if OUT.exists():
    for line in open(OUT):
        try: done.add(json.loads(line)["video_id"])
        except Exception: pass

todo = []
for a in ("democrats", "whitehouse", "republicans"):
    d = DESC / a
    if not d.exists(): continue
    for f in sorted(d.glob("*.json")):
        pid = f.stem.replace("_carousel", "")
        if pid not in done: todo.append((a, pid, f))
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if limit: todo = todo[:limit]
print(f"to classify: {len(todo)}")

tot_cost = 0.0
with open(OUT, "a") as out:
    for a, pid, f in todo:
        desc = json.dumps(json.load(open(f)), ensure_ascii=False)[:8000]
        msg = client.messages.create(
            model=MODEL, max_tokens=200, temperature=0,
            messages=[{"role": "user", "content": PROMPT.format(desc=desc, cap=caps.get(pid, ""))}])
        text = msg.content[0].text
        m = re.search(r"\{.*\}", text, re.S)
        row = json.loads(m.group(0)) if m else {"error": text[:200]}
        row.update({"video_id": pid, "account": a})
        cost = msg.usage.input_tokens/1e6*PRICE_IN + msg.usage.output_tokens/1e6*PRICE_OUT
        tot_cost += cost
        out.write(json.dumps(row) + "\n"); out.flush()
        print(f"  {pid}  {row.get('attack','?')}/{row.get('crudeness','?')}  in={msg.usage.input_tokens} out={msg.usage.output_tokens}  ${cost:.4f}")
        time.sleep(1)
print(f"done · total classify cost ${tot_cost:.4f}")
