"""
01_openai_client.py
-------------------
Talk to your private expert model using the official OpenAI Python SDK.

The point of this file: you do not write a custom client for your model.
Both Ollama and vLLM expose an OpenAI-compatible /v1 endpoint, so the same
code that talked to gpt-4 talks to your 4.5 GB medical Gemma running on a
laptop. You change two lines, base_url and model, and nothing else.

Run against Ollama (default):
    ollama serve                       # in another terminal
    python 01_openai_client.py

Run against vLLM instead:
    python 01_openai_client.py --base http://localhost:8000/v1 \
        --model ./medical-gemma-merged

Streaming and non-streaming are both shown; streaming is what you want in a
UI so the user sees tokens as they are produced instead of waiting for the
whole answer.
"""
import argparse
from openai import OpenAI


def ask(client, model, question, stream=True):
    messages = [{"role": "user", "content": question}]

    if not stream:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=512,
        )
        return resp.choices[0].message.content

    # Streaming: print tokens as they arrive.
    out = []
    with client.chat.completions.stream(
        model=model, messages=messages, temperature=0.3, max_tokens=512,
    ) as s:
        for event in s:
            if event.type == "content.delta":
                print(event.delta, end="", flush=True)
                out.append(event.delta)
        print()
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    # Ollama's OpenAI-compatible endpoint. For vLLM use http://localhost:8000/v1
    ap.add_argument("--base", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="medical-gemma")
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args()

    # api_key is required by the SDK but ignored by a local server. Send anything.
    client = OpenAI(base_url=args.base, api_key="not-needed-for-local")

    question = "What are the first-line treatments for type 2 diabetes?"
    print(f"# {args.model} @ {args.base}\n")
    ask(client, args.model, question, stream=not args.no_stream)


if __name__ == "__main__":
    main()
