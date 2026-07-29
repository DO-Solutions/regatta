#!/usr/bin/env python3
"""Webinar suite run — K2.6 vs K3 vs Opus 5 across the full eval set.

Webinar ground rule: 429s are NOT a model quality signal, so any question
that fails on platform pressure (429/5xx after in-call retries) is retried in later
sweeps until it completes cleanly; only clean completions enter the statistics, and
the excluded/retried count is reported honestly next to the results.

Every arm gets the IDENTICAL prompt: shared top-5 retrieval from Poseidon's Weaviate
KB + the byte-identical system message (SYSMSG — no per-model persona differences).
Judge = an open-source THIRD-family model (deepseek-v4-pro on DO by default): neither
Kimi nor Claude is graded by its own vendor's model.

Usage:  set -a; . ./.env; set +a; python3 suite_runner.py --out suite_results.json

Outputs <out> (full per-question records) and prints the aggregate table +
cost-at-scale extrapolation for the closing slide.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

DO_INFER = "https://inference.do-ai.run/v1"
ANTH = "https://api.anthropic.com/v1/messages"
CFG = {k: os.environ.get(k, "") for k in ("DO_KEY", "ANTHROPIC_KEY", "RAG_KEY")}
OPUS_BACKEND = os.environ.get("OPUS_BACKEND", "do")

from race_server import (MODELS, SYSMSG, judge_one,  # single source of truth
                         retrieve)                    # shared retrieval (live, cache, or none)


def _post(url, body, headers, timeout=300, tries=6):
    """Returns (status, json). Retries 429/5xx in-call; caller sweeps leftovers."""
    for i in range(tries):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")[:200]
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(min(45, 5 * (i + 1)))
                continue
            return e.code, {"_err": txt}
        except Exception as e:
            if i < tries - 1:
                time.sleep(min(30, 4 * (i + 1)))
                continue
            return -1, {"_err": repr(e)[:200]}
    return -1, {"_err": "retries exhausted"}


def degenerate(text):
    """Serving-fault detector: a response that is mostly one repeated character (observed
    live 2026-07-29: K3 under launch-day load returning 4096 tokens of '!'). Platform
    failure, not model quality — treated like a 429: excluded and re-run."""
    t = (text or "").strip()
    if len(t) < 20:
        return False
    return max(t.count(c) for c in set(t[:200])) / len(t) > 0.5


def ask(arm, q):
    m = MODELS[arm]
    ctx = "\n\n---\n\n".join(retrieve(q))[:12000]
    t0 = time.time()
    if arm == "opus" and OPUS_BACKEND == "anthropic":
        st, d = _post(ANTH, {"model": m["anthropic_id"], "max_tokens": 4096,
                             "system": SYSMSG + "\n\nCONTEXT:\n" + ctx,
                             "messages": [{"role": "user", "content": q}]},
                      {"x-api-key": CFG["ANTHROPIC_KEY"],
                       "anthropic-version": "2023-06-01"})
        if st != 200:
            return {"ok": False, "status": st, "err": str(d)[:160],
                    "latency_s": round(time.time() - t0, 2)}
        u = d.get("usage") or {}
        pt, ct = u.get("input_tokens") or 0, u.get("output_tokens") or 0
        text = "".join(b.get("text", "") for b in d.get("content", []))
    else:
        # 4096: reasoning tokens share this budget — at 1024, 24/57 K3 answers truncated
        # and scored 33 vs 92 uncapped (measured 2026-07-29). Generous is correct.
        body = {"model": m["do_id"], "max_tokens": 4096,
                "messages": [{"role": "system", "content": SYSMSG + "\n\nCONTEXT:\n" + ctx},
                             {"role": "user", "content": q}]}
        if "top_p" in m:
            body["top_p"] = m["top_p"]
        st, d = _post(DO_INFER + "/chat/completions", body,
                      {"Authorization": "Bearer " + CFG["DO_KEY"]})
        if st != 200:
            return {"ok": False, "status": st, "err": str(d)[:160],
                    "latency_s": round(time.time() - t0, 2)}
        u = d.get("usage") or {}
        pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
        text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if degenerate(text):
        return {"ok": False, "status": "degenerate", "err": "degenerate output (repeated-char spam)",
                "latency_s": round(time.time() - t0, 2)}
    pin, pout = m["price"]
    return {"ok": True, "latency_s": round(time.time() - t0, 2), "text": text,
            "prompt_tokens": pt, "completion_tokens": ct,
            "cost_usd": round(pt * pin / 1e6 + ct * pout / 1e6, 6)}


judge = judge_one  # open-source third-family judge (deepseek-v4-pro), shared with the UI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="k26,k3,opus")
    ap.add_argument("--dataset", default=os.environ.get("DATASET", "evalset.csv"))
    ap.add_argument("--out", default="suite_results.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sweeps", type=int, default=3,
                    help="extra passes to re-run 429-excluded questions")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    rows = list(csv.DictReader(open(a.dataset)))
    if a.limit:
        rows = rows[:a.limit]

    results = {arm: [None] * len(rows) for arm in arms}
    retried = {arm: 0 for arm in arms}

    for sweep in range(1 + a.sweeps):
        pending = [(arm, i) for arm in arms for i in range(len(rows))
                   if results[arm][i] is None or not results[arm][i].get("ok")]
        if not pending:
            break
        if sweep:
            print(f"\n-- sweep {sweep}: re-running {len(pending)} platform-failed calls --")
            for arm, _ in pending:
                retried[arm] += 1
            time.sleep(20)
        for n, (arm, i) in enumerate(pending, 1):
            q, ref = rows[i]["query"], rows[i]["expected_response"]
            res = ask(arm, q)
            if res.get("ok"):
                res["judge"] = judge(q, ref, res["text"])
            res["query"] = q
            results[arm][i] = res
            c = (res.get("judge") or {}).get("correctness")
            print(f"[s{sweep} {n}/{len(pending)}] {arm} #{i} {res.get('latency_s')}s "
                  f"score={c} {'OK' if res.get('ok') else 'ERR ' + str(res.get('err'))[:50]}",
                  flush=True)
            json.dump(results, open(a.out, "w"), indent=1)

    print("\n=== AGGREGATE (clean completions only; 429s re-run, not scored) ===")
    print(f"{'arm':<6} {'n':>3} {'corr':>6} {'faith':>6} {'lat':>7} {'p95':>7} "
          f"{'$/ans':>9} {'total$':>8} {'excl':>5}")
    scale = {}
    for arm in arms:
        ok = [r for r in results[arm] if r and r.get("ok")]
        excl = len(rows) - len(ok)
        sc = [r["judge"]["correctness"] for r in ok
              if (r.get("judge") or {}).get("correctness") is not None]
        fa = [r["judge"]["faithfulness"] for r in ok
              if (r.get("judge") or {}).get("faithfulness") is not None]
        lat = sorted(r["latency_s"] for r in ok)
        cost = [r["cost_usd"] for r in ok]
        mean_cost = statistics.mean(cost) if cost else 0
        scale[arm] = mean_cost
        if not (ok and sc):
            print(f"{arm:<6} no scored data ({excl} excluded)")
            continue
        print(f"{arm:<6} {len(ok):>3} "
              f"{statistics.mean(sc):>6.1f} {statistics.mean(fa) if fa else 0:>6.1f} "
              f"{statistics.mean(lat):>6.1f}s {lat[int(.95 * (len(lat) - 1))]:>6.1f}s "
              f"{mean_cost:>9.5f} {sum(cost):>8.4f} {excl:>5}")

    print("\n=== COST AT SCALE (per month) ===")
    print(f"{'arm':<6} {'10k q/mo':>12} {'100k q/mo':>12} {'1M q/mo':>12}")
    for arm in arms:
        c = scale.get(arm, 0)
        print(f"{arm:<6} {c * 1e4:>11.0f}$ {c * 1e5:>11.0f}$ {c * 1e6:>11.0f}$")
    print(f"\nfull records: {a.out}")


if __name__ == "__main__":
    required = ["DO_KEY", "RAG_KEY"] + (["ANTHROPIC_KEY"] if OPUS_BACKEND == "anthropic" else [])
    missing = [k for k in required if not CFG.get(k)]
    if missing:
        sys.exit(f"missing env: {missing} — use run.sh")
    main()
