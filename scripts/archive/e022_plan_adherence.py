"""E022: does Gemma follow plans, and where do turns and time go? (learning vs fixes, v2+v3)"""
import json, re, statistics as st
from collections import Counter
from pathlib import Path
from worldmodel_harness.fixes import endpoint

ROOT = Path("runs/e022-learning-stream/trajectories")
HEADER = "[Experience] Earlier runs"

def calls_of(msg):
    out = []
    for b in msg.get("blocks") or []:
        if b.get("kind") == "tool_call":
            try: a = json.loads(b["arguments"])
            except Exception: a = {}
            out.append((b["name"], a if isinstance(a, dict) else {}))
    return out

def planned(line):
    m = re.match(r"\d+\. (\w+) (\S+)", line)
    url = re.sub(r"<[^>]*>", "{x}", m.group(2))
    return endpoint({"method": m.group(1), "url": url})

def trial(path):
    groups = json.loads((path / "trajectory.json").read_text())
    msgs = [m for g in groups for m in g if isinstance(m, dict)]
    res = json.loads((path / "result.json").read_text())
    s = Counter()
    s["model_s"] = sum(c.get("elapsed_seconds") or 0 for c in res.get("model_call_costs", []))
    s["out_tok"] = (res.get("token_usage") or {}).get("answer", 0)
    events, pending = [], None
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            cs = calls_of(m)
            s["turns"] += 1
            for name, a in cs:
                s[name] += 1
            if pending is not None and cs:
                names = {n for n, _ in cs}
                fetched = [endpoint(a) for n, a in cs if n == "api_fetch"]
                if pending[0] in fetched: kind = "followed step 1"
                elif any(p in fetched for p in pending[1:]): kind = "took later step"
                elif fetched: kind = "other fetch"
                elif "api_search" in names: kind = "searched again"
                else: kind = "/".join(sorted(names))
                events.append(kind); pending = None
        elif m.get("role") == "tool" and HEADER in str(m.get("content")):
            lines = str(m["content"]).split(HEADER, 1)[1].split("\n")
            pending = [planned(l) for l in lines if re.match(r"\d+\. ", l)]
    return s, events, res

def mean(xs): return st.mean(xs) if xs else float("nan")
for rnd in ("~v2", "~v3"):
    tasks = sorted(p.name for p in ROOT.iterdir() if p.name.endswith(rnd))
    ev, rows = Counter(), {"fixes": [], "learning": [], "learning_planned": [], "fixes_planned": []}
    for t in tasks:
        L, e, _ = trial(ROOT / t / "learning-r0"); F, _, _ = trial(ROOT / t / "fixes-r0")
        ev.update(e)
        rows["learning"].append(L); rows["fixes"].append(F)
        if e: rows["learning_planned"].append(L); rows["fixes_planned"].append(F)
    print(f"== {rnd}  plan reactions: {dict(ev.most_common())}  (n={sum(ev.values())})")
    for k, rs in rows.items():
        print(f"  {k:17s} n={len(rs):2d} turns={mean([r['turns'] for r in rs]):.1f} search={mean([r['api_search'] for r in rs]):.1f} "
              f"fetch={mean([r['api_fetch'] for r in rs]):.1f} model_s={mean([r['model_s'] for r in rs]):.0f} out_tok={mean([r['out_tok'] for r in rs]):.0f} "
              f"s/turn={mean([r['model_s']/max(r['turns'],1) for r in rs]):.2f}")
