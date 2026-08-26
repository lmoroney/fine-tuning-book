"""
06_fastapi_gateway.py
---------------------
A thin production gateway in front of your model server.

Ollama and vLLM already speak OpenAI. So why put anything in front? Because
the last mile of serving is not inference, it is the boring operational glue:
an auth check so not everyone on the network can hit your model, a health
endpoint your load balancer can poll, a request timeout so one slow call can't
pin a worker forever, and a single place to add logging and rate limits.

This gateway forwards chat requests to your backend (Ollama or vLLM) and adds
those concerns without re-implementing inference. Crucially, it moves BYTES,
not parsed JSON: the response is piped through as the backend produces it, so
token-by-token streaming survives the trip. A gateway that called resp.json()
would buffer the whole answer, turning time-to-first-token into
time-to-last-token and breaking "stream": true requests entirely.

    pip install fastapi uvicorn httpx
    BACKEND=http://localhost:8000/v1 API_KEY=secret uvicorn 06_fastapi_gateway:app --port 9000
"""
import os
import time
import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

BACKEND = os.environ.get("BACKEND", "http://localhost:11434/v1")
API_KEY = os.environ.get("API_KEY", "")           # empty disables auth

# Timeouts need care with streaming: CONNECT_TIMEOUT guards a dead backend,
# but the read timeout must be long enough that a healthy long generation
# doesn't get killed mid-stream. Tune READ_TIMEOUT to your slowest expected
# generation, not your fastest.
CONNECT_TIMEOUT = float(os.environ.get("CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.environ.get("READ_TIMEOUT", "120"))

app = FastAPI(title="expert-gateway")
_client = httpx.AsyncClient(
    base_url=BACKEND,
    timeout=httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT,
                          write=10.0, pool=5.0),
)


@app.get("/health")
async def health():
    # Cheap liveness check for a load balancer. Confirms the backend answers.
    try:
        r = await _client.get("/models")
        return {"status": "ok" if r.status_code == 200 else "degraded"}
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="backend unreachable")


@app.post("/v1/chat/completions")
async def chat(request: Request, authorization: str = Header(default="")):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid api key")

    body = await request.body()
    t0 = time.perf_counter()
    try:
        # Open the backend request in streaming mode. We get the response
        # headers here, but the body is NOT read yet -- it flows through the
        # StreamingResponse below as the model produces it.
        upstream_req = _client.build_request(
            "POST", "/chat/completions", content=body,
            headers={"Content-Type": "application/json"},
        )
        upstream = await _client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="model timed out")
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="backend unreachable")

    ttfb_ms = (time.perf_counter() - t0) * 1000
    # One structured log line per request is the foundation of observability.
    # With streaming we log time-to-first-byte; total duration would require
    # waiting for the stream to finish, which is exactly what we're avoiding.
    print(f'{{"path":"/v1/chat/completions","status":{upstream.status_code},'
          f'"ttfb_ms":{ttfb_ms:.0f}}}')

    # Pipe the response through untouched -- works identically for a single
    # JSON body and for an SSE token stream, because both are just bytes.
    # BackgroundTask closes the upstream connection after the last byte.
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        background=BackgroundTask(upstream.aclose),
    )
