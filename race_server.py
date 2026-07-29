#!/usr/bin/env python3
"""Webinar demo server — three Poseidons race the same question (2026-07-29 K3 webinar).

Serves race.html and streams three model arms side-by-side over SSE, each assembled
EXACTLY like the eval's control arm: retrieve top-5 from Poseidon's Weaviate KB (via
rag-api), inject as context, ask the model. Retrieval runs ONCE per question and is
shared by all arms — "same knowledge, three brains".

  GET  /            → race.html
  GET  /questions   → the eval dataset as JSON [{qid, query, expected_response}]
  GET  /stream?arm=k3|k26|opus&q=<text>   → SSE: {"t":"think"|"ans","d":...} deltas,
                                            then {"t":"done","stats":{...}}
  POST /judge  {"q","ref","answers":{arm:text}} → per-arm correctness/faithfulness
                                            (JUDGE_MODEL, default deepseek-v4-pro on DO —
                                            an open-source THIRD-family judge: neither
                                            candidate is graded by its own vendor)

Env (copy .env.example to .env — never hardcode keys in code):
  DO_KEY         DO GenAI serverless inference key (all-models, 2026-07-29)
  ANTHROPIC_KEY  Anthropic API key — ONLY needed when OPUS_BACKEND=anthropic
  RAG_URL        optional retrieval endpoint (POST /search {query, top_k} -> {results:[{text}]});
                 unset = context-free comparison, or use RAG_CACHE for precomputed chunks
  RAG_KEY        rag-api key
  OPUS_BACKEND   "do" (default — all arms on DigitalOcean) or "anthropic" to route the
                 Opus arm via the Anthropic API (needs ANTHROPIC_KEY)
  DATASET        eval CSV (default /tmp/evalset.csv)
  PORT           default 8130

Python stdlib only — no dependencies.
"""
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DO_INFER = "https://inference.do-ai.run/v1"
ANTH = "https://api.anthropic.com/v1/messages"

# $/Mtok (in, out) — all three VERIFIED 2026-07-29 against the DO API's own model
# records (GET /v2/gen-ai/models → pricing.{input,output}_price_per_million × 1e6).
MODELS = {
    "k3":   {"label": "Kimi K3",       "do_id": "kimi-k3",   "price": (3.0, 15.0),
             "top_p": 0.95},   # K3 400s on any top_p other than 0.95 IF sent
    "k26":  {"label": "Kimi K2.6",     "do_id": "kimi-k2.6", "price": (0.76, 3.20)},
    "opus": {"label": "Claude Opus 5", "do_id": "anthropic-claude-opus-5",
             "anthropic_id": "claude-opus-5", "price": (5.0, 25.0)},
}

SYSMSG = ("You are Poseidon, a DigitalOcean solutions expert. Answer the question using the "
          "context when it is relevant. Be concise and concrete. If the context does not "
          "cover it, answer from your own knowledge and say so briefly.")

JUDGE_PROMPT = """You are grading an answer against a reference answer.

QUESTION: {q}

REFERENCE (ground truth): {ref}

CANDIDATE ANSWER: {ans}

Score the CANDIDATE on two axes, 0-100 each:
- correctness: is it factually right and does it actually answer the question? Extra correct
  detail is fine. Being right but phrased differently from the reference is still correct.
- faithfulness: does it agree with the reference, without contradicting it or inventing
  specifics the reference does not support?

Reply with ONLY a JSON object: {{"correctness": <int>, "faithfulness": <int>, "why": "<12 words>"}}"""

CFG = {k: os.environ.get(k, "") for k in ("DO_KEY", "ANTHROPIC_KEY", "RAG_KEY")}
RAG_URL = os.environ.get("RAG_URL", "")   # empty = no live retriever (cache or context-free)
OPUS_BACKEND = os.environ.get("OPUS_BACKEND", "do")
DATASET = os.environ.get("DATASET", "evalset.csv")

_rag_cache, _rag_lock = {}, threading.Lock()
# Optional precomputed retrieval (question -> [chunks]): lets the demo run with byte-
# identical chunks where the live retriever isn't reachable (and makes runs reproducible).
RAG_CACHE = os.environ.get("RAG_CACHE", "")
if RAG_CACHE and os.path.exists(RAG_CACHE):
    _rag_cache.update(json.load(open(RAG_CACHE)))


def retrieve(q):
    """Top-5 chunks, cached per question so every arm sees identical context.
    No RAG_URL configured -> context-free comparison (chunks=[]), by design."""
    with _rag_lock:
        if q in _rag_cache:
            return _rag_cache[q]
    if not RAG_URL:
        return []
    req = urllib.request.Request(
        RAG_URL + "/search",
        data=json.dumps({"query": q, "collection": "*", "top_k": 5}).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": CFG["RAG_KEY"]})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            chunks = [x.get("text", "") for x in json.load(r).get("results", [])]
    except Exception as e:
        print("retrieve failed:", repr(e)[:120], file=sys.stderr)
        chunks = []
    with _rag_lock:
        _rag_cache[q] = chunks
    return chunks


def _sse(w, obj):
    w.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
    w.flush()


def _iter_sse_lines(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data: "):
            yield line[6:]


def stream_do(w, arm, q, chunks):
    """DO serverless: thinking arrives in delta.reasoning_content, answer in delta.content."""
    m = MODELS[arm]
    ctx = "\n\n---\n\n".join(chunks)[:12000]
    # 2048: reasoning tokens share the completion budget — at 1024 both Kimi arms hit the
    # cap exactly (K2.6 spent ~3.5k chars thinking), truncating answers mid-sentence on camera
    body = {"model": m["do_id"], "max_tokens": 4096, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": SYSMSG + "\n\nCONTEXT:\n" + ctx},
                         {"role": "user", "content": q}]}
    if "top_p" in m:
        body["top_p"] = m["top_p"]
    req = urllib.request.Request(DO_INFER + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + CFG["DO_KEY"]})
    t0, tft, tfa, usage = time.time(), None, None, {}
    n_think = n_ans = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for data in _iter_sse_lines(resp):
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except ValueError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                rc = delta.get("reasoning_content")
                c = delta.get("content")
                if rc:
                    tft = tft or time.time()
                    n_think += len(rc)
                    _sse(w, {"t": "think", "d": rc})
                if c:
                    tfa = tfa or time.time()
                    n_ans += len(c)
                    _sse(w, {"t": "ans", "d": c})
    return _finish(w, m, t0, tft, tfa, n_think, n_ans, usage.get("prompt_tokens"),
                   usage.get("completion_tokens"))


def stream_anthropic(w, arm, q, chunks):
    """Anthropic messages API with extended thinking, so the audience sees Opus think too."""
    m = MODELS[arm]
    ctx = "\n\n---\n\n".join(chunks)[:12000]
    # Claude 5 family: extended thinking is "adaptive" (the old enabled+budget shape 400s)
    body = {"model": m["anthropic_id"], "max_tokens": 3000, "stream": True,
            "thinking": {"type": "adaptive"},
            "system": SYSMSG + "\n\nCONTEXT:\n" + ctx,
            "messages": [{"role": "user", "content": q}]}
    req = urllib.request.Request(ANTH, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "x-api-key": CFG["ANTHROPIC_KEY"],
                                          "anthropic-version": "2023-06-01"})
    t0, tft, tfa = time.time(), None, None
    n_think = n_ans = 0
    pt = ct = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for data in _iter_sse_lines(resp):
            try:
                d = json.loads(data)
            except ValueError:
                continue
            typ = d.get("type")
            if typ == "message_start":
                pt = (d.get("message", {}).get("usage") or {}).get("input_tokens")
            elif typ == "content_block_delta":
                delta = d.get("delta") or {}
                if delta.get("type") == "thinking_delta":
                    tft = tft or time.time()
                    n_think += len(delta.get("thinking", ""))
                    _sse(w, {"t": "think", "d": delta.get("thinking", "")})
                elif delta.get("type") == "text_delta":
                    tfa = tfa or time.time()
                    n_ans += len(delta.get("text", ""))
                    _sse(w, {"t": "ans", "d": delta.get("text", "")})
            elif typ == "message_delta":
                ct = (d.get("usage") or {}).get("output_tokens") or ct
    return _finish(w, m, t0, tft, tfa, n_think, n_ans, pt, ct)


def _finish(w, m, t0, tft, tfa, n_think, n_ans, pt, ct):
    now = time.time()
    est = pt is None or ct is None
    pt = pt if pt is not None else 0
    ct = ct if ct is not None else int((n_think + n_ans) / 4)  # ~4 chars/token estimate
    pin, pout = m["price"]
    stats = {"ttfr_s": round((tft or now) - t0, 2),        # first THINKING token
             "ttfa_s": round((tfa or now) - t0, 2),        # first ANSWER token
             "total_s": round(now - t0, 2),
             "think_chars": n_think, "ans_chars": n_ans,
             "prompt_tokens": pt, "completion_tokens": ct, "tokens_estimated": est,
             "cost_usd": round(pt * pin / 1e6 + ct * pout / 1e6, 6),
             "tok_per_s": round(ct / max(0.1, now - (tfa or tft or t0)), 1)}
    _sse(w, {"t": "done", "stats": stats})
    return stats


# Judge: an OPEN-SOURCE model from a THIRD family (not Moonshot, not Anthropic), so no
# candidate is judged by its own vendor's model — kills the self-preference caveat outright.
# max_tokens 2000: the old 700 budget ran dry on long candidates and produced unparseable
# verdicts (the JSONDecodeError that surfaced raw in the UI grade cards).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-pro")


def _extract_json(txt):
    txt = txt.strip()
    if txt.startswith("```"):                     # strip markdown fences if the model adds them
        txt = txt.strip("`")
        if txt.startswith("json"):
            txt = txt[4:]
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in judge reply")
    return json.loads(txt[i:j + 1])


def judge_one(q, ref, ans, tries=3):
    if not (ans or "").strip():
        return {"correctness": 0, "faithfulness": 0, "why": "empty answer"}
    body = {"model": JUDGE_MODEL, "max_tokens": 2000,
            "messages": [{"role": "user",
                          "content": JUDGE_PROMPT.format(q=q, ref=ref, ans=ans[:4000])}]}
    last = "judge failed"
    for i in range(tries):
        req = urllib.request.Request(DO_INFER + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + CFG["DO_KEY"]})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            return _extract_json(txt)
        except Exception as e:
            last = f"judge retry exhausted ({type(e).__name__})"
            time.sleep(4 * (i + 1))
    return {"correctness": None, "faithfulness": None, "why": last}


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # quiet
        pass

    def _hdr(self, code, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "race.html"), "rb").read()
            self._hdr(200, "text/html; charset=utf-8")
            self.wfile.write(page)
        elif u.path == "/questions":
            rows = [{"qid": i, **r} for i, r in
                    enumerate(csv.DictReader(open(DATASET)))]
            self._hdr(200, "application/json")
            self.wfile.write(json.dumps(rows).encode())
        elif u.path == "/stream":
            qs = parse_qs(u.query)
            arm = (qs.get("arm") or [""])[0]
            q = (qs.get("q") or [""])[0]
            if arm not in MODELS or not q.strip():
                self._hdr(400, "text/plain")
                self.wfile.write(b"need arm=k3|k26|opus and q=")
                return
            self._hdr(200, "text/event-stream",
                      {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            try:
                chunks = retrieve(q)
                _sse(self.wfile, {"t": "ctx", "n": len(chunks)})
                if arm == "opus" and OPUS_BACKEND == "anthropic":
                    stream_anthropic(self.wfile, arm, q, chunks)
                else:
                    stream_do(self.wfile, arm, q, chunks)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                _sse(self.wfile, {"t": "err", "d": f"HTTP {e.code}: {detail}"})
            except Exception as e:
                try:
                    _sse(self.wfile, {"t": "err", "d": repr(e)[:200]})
                except Exception:
                    pass
        else:
            self._hdr(404, "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/judge":
            self._hdr(404, "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        out = {arm: judge_one(body["q"], body["ref"], ans)
               for arm, ans in (body.get("answers") or {}).items()}
        self._hdr(200, "application/json")
        self.wfile.write(json.dumps(out).encode())


if __name__ == "__main__":
    required = ["DO_KEY"] + (["ANTHROPIC_KEY"] if OPUS_BACKEND == "anthropic" else [])
    if RAG_URL:
        required.append("RAG_KEY")
    missing = [k for k in required if not CFG.get(k)]
    if missing:
        sys.exit(f"missing env: {missing} — see .env.example")
    if not (RAG_URL or _rag_cache):
        print("note: no RAG_URL and no RAG_CACHE — running context-free (no retrieval)")
    port = int(os.environ.get("PORT", "8130"))
    print(f"race server on http://0.0.0.0:{port}  (opus backend: {OPUS_BACKEND})")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
