"""E022 per-round arms and paired learning-vs-fixes/baseline comparisons. Usage: python scripts/e022_summary.py"""
import json, statistics as st, sys
rows = json.load(open("runs/e022-learning-stream/summary.json"))["results"]
by = {(r["task"], r["arm"]): r for r in rows}
V = lambda r: int(r["task"].rsplit("~v",1)[1]) - 1
for r in rows: r["round"] = V(r)
rounds = sorted({r["round"] for r in rows})
def show(name, rs):
    t = [r["duration_seconds"] for r in rs]
    print(f"  {name:9s} n={len(rs)} strict={sum(r['strict_eval'] for r in rs):2d} partial={st.mean(r['partial_credit'] for r in rs):.3f} "
          f"med_t={st.median(t):.0f} mean_t={st.mean(t):.0f} env={st.mean(r['environment_calls'] for r in rs):.1f} "
          f"in_tok={st.mean(r['input_tokens'] or 0 for r in rs)/1000:.0f}k model_calls={st.mean(r['model_calls'] for r in rs):.1f} "
          f"err={sum(bool(r['error_type']) for r in rs)}" + (f" with_plan={sum(r['plans_shown']>0 for r in rs)}" if name=='learning' else ""))
for rnd in rounds:
    print(f"round v{rnd+1}")
    tasks = sorted({r["task"] for r in rows if r["round"] == rnd})
    for arm in ("baseline", "fixes", "learning"):
        show(arm, [by[t, arm] for t in tasks if (t, arm) in by])
    for other in ("fixes", "baseline"):
        for subset in ("all", "plans"):
            ts = [t for t in tasks if (t, "learning") in by and (t, other) in by and (subset == "all" or by[t, "learning"]["plans_shown"] > 0)]
            if not ts: continue
            L = [by[t, "learning"] for t in ts]; O = [by[t, other] for t in ts]
            dp = [a["partial_credit"] - b["partial_credit"] for a, b in zip(L, O)]
            ds = [a["strict_eval"] - b["strict_eval"] for a, b in zip(L, O)]
            dt = [a["duration_seconds"] - b["duration_seconds"] for a, b in zip(L, O)]
            de = [a["environment_calls"] - b["environment_calls"] for a, b in zip(L, O)]
            print(f"   learning-{other:8s} [{subset:5s} n={len(ts):2d}] strict {sum(ds):+d} (win {sum(x>0 for x in ds)}/loss {sum(x<0 for x in ds)}) "
                  f"partial {st.mean(dp):+.3f} (better {sum(x>1e-9 for x in dp)}, worse {sum(x<-1e-9 for x in dp)}) "
                  f"time med {st.median(dt):+.0f}s mean {st.mean(dt):+.0f}s faster {sum(x<0 for x in dt)}/{len(dt)} env mean {st.mean(de):+.1f}")
