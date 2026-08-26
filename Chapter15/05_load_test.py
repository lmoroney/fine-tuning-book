"""
05_load_test.py
---------------
Measure what your serving stack actually delivers under concurrency.

Two numbers decide whether a deployment is viable: per-request latency (how
long one user waits) and aggregate throughput (how many tokens/sec the server
sustains across all users). A single-user test tells you the first and hides
the second. This script fires N concurrent requests and reports both, so you
can compare Ollama vs vLLM on YOUR hardware instead of trusting a blog post.

    python 05_load_test.py --base http://localhost:8000/v1 \
        --model medical-gemma --concurrency 16 --requests 64
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

PROMPTS = [
    "What are the first-line treatments for type 2 diabetes?",
    "Explain the mechanism of action of beta blockers.",
    "When is a CT scan preferred over an MRI?",
    "List the common side effects of metformin.",
]


def one_request(client, model, prompt):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=256,
    )
    dt = time.perf_counter() - t0
    toks = resp.usage.completion_tokens if resp.usage else 0
    return dt, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="medical-gemma")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=64)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base, api_key="not-needed")

    latencies, total_tokens = [], 0
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [
            pool.submit(one_request, client, args.model, PROMPTS[i % len(PROMPTS)])
            for i in range(args.requests)
        ]
        for f in as_completed(futs):
            dt, toks = f.result()
            latencies.append(dt)
            total_tokens += toks
    wall = time.perf_counter() - wall_start

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    print(f"requests={args.requests} concurrency={args.concurrency}")
    print(f"wall time         {wall:6.2f} s")
    print(f"latency p50/p95   {p50:6.2f} / {p95:6.2f} s")
    print(f"throughput        {total_tokens / wall:7.1f} output tok/s")


if __name__ == "__main__":
    main()
