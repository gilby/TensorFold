"""V100 bench client: 437, drafted == draft:false, serial == N=2/N=4, single-stream and N=2/4/8 tok/s, prefill."""
import concurrent.futures as cf, glob, hashlib, json, os, sys, time, urllib.request, uuid

base, model = sys.argv[1], sys.argv[2]
parts = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else {"arith", "exact", "single", "conc", "prefill"}
U = base + "/chat/completions"


def call(msgs, mt=512, think=False, extra=None):
    b = {"model": model, "messages": msgs, "max_tokens": mt, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": think}}
    b.update(extra or {})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(),
                                                                {"Content-Type": "application/json"}), timeout=7200))
    return r, time.perf_counter() - t


def text(r):
    m = r["choices"][0]["message"]
    return (m.get("content") or "") + "\x00" + (m.get("reasoning_content") or m.get("reasoning") or "")


P = ["Write a Python function that parses an ISO-8601 duration like P3DT4H5M into seconds, with tests.",
     "Explain, for a hospital pharmacist, how to dose enoxaparin in a patient with CrCl 25 mL/min.",
     "Write a bash script that rotates logs in /var/log/app older than 7 days and gzips them.",
     "Summarise the causes of the 1929 stock market crash in five bullet points.",
     "Write a C function that reverses a singly linked list in place, and explain its complexity.",
     "Describe how a TCP three-way handshake works and what SYN flooding exploits.",
     "Write a SQL query that finds the second highest salary per department, with an example schema.",
     "Explain the difference between a mutex and a semaphore with a short Rust example."]
out = {}
call([{"role": "user", "content": "hi"}], 8)
if "arith" in parts:
    r, dt = call([{"role": "user", "content": "What is 19 times 23? Reply with only the number."}], 64)
    out["arith_nothink"] = r["choices"][0]["message"].get("content", "").strip()
    r, dt = call([{"role": "user", "content": "What is 19 times 23? Reply with only the number."}], 4096, think=True)
    out["arith_think"] = {"content": (r["choices"][0]["message"].get("content") or "").strip()[:40],
                          "tokens": r["usage"]["completion_tokens"], "s": round(dt, 1)}


def one(p, mt=512, extra=None):
    r, dt = call([{"role": "user", "content": p}], mt, extra=extra)
    return hashlib.sha256(text(r).encode()).hexdigest()[:16], r["usage"]["completion_tokens"], dt


if "exact" in parts:
    ex = []
    for p in P[:3]:
        a, b = one(p, 256), one(p, 256, {"draft": False})
        ex.append({"drafted": a[0], "serial": b[0], "equal": a[0] == b[0], "drafted_tok_s": round(a[1] / a[2], 1),
                   "serial_tok_s": round(b[1] / b[2], 1)})
    out["drafted_vs_serial"] = ex
ser = None
if "single" in parts:                     # fresh prompts (a unique nonce each): no prompt-cache hit
    tag = uuid.uuid4().hex[:8]
    fresh = [one(f"[{tag}-{i}] {p}") for i, p in enumerate(P[:4])]
    out["single_tok_s"] = [round(n / d, 1) for _, n, d in fresh]
    out["single_mean"] = round(sum(n for _, n, _ in fresh) / sum(d for _, _, d in fresh), 1)
    out["single_tokens"] = [n for _, n, _ in fresh]
if "conc" in parts:
    with cf.ThreadPoolExecutor(8) as pool:          # warm the concurrent decoder's kernels
        list(pool.map(lambda p: one(p, 16), P))
    tag = uuid.uuid4().hex[:8]
    Q = [f"[{tag}-{i}] {p}" for i, p in enumerate(P)]
    ser = [one(p) for p in Q]
    out["serial_tok_s"] = [round(n / d, 1) for _, n, d in ser]
    for n in (2, 4, 8):
        t = time.perf_counter()
        with cf.ThreadPoolExecutor(n) as pool:
            got = list(pool.map(one, Q[:n]))
        w = time.perf_counter() - t
        out[f"n{n}_agg_tok_s"] = round(sum(x[1] for x in got) / w, 1)
        out[f"n{n}_equal_serial"] = all(a[0] == b[0] for a, b in zip(ser, got))
if "prefill" in parts:
    here = os.path.dirname(os.path.abspath(__file__))
    src = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(here, "../src/tensorfold/server/*.py"))))
    for chars in (26500,):
        body = f"[{uuid.uuid4().hex}]\n{src[:chars]}\n\nName one class defined above."
        r, dt = call([{"role": "user", "content": body}], 1)
        out["prefill"] = {"prompt_tokens": r["usage"]["prompt_tokens"], "wall_s": round(dt, 2),
                          "tok_s": round(r["usage"]["prompt_tokens"] / dt)}
print(json.dumps(out, indent=1))
