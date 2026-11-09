# Contract Demo

This demo runs a deterministic fake agent and checks three behaviors:

1. A read-only request returns `success` with a `get_my_tasks` tool call.
2. A write request returns `pending_confirmation` and a confirmation token; it does not claim success.
3. A cross-department request is refused without a tool call.

It is a contract demo, not a model quality baseline. It uses no Ollama, Qdrant, MySQL, Redis, or Java backend.

## Run with Docker

```bash
docker compose -f demo/docker-compose.yml up --build --abort-on-container-exit --exit-code-from smoke
```

Expected output:

```text
[PASS] read-only query -> success + get_my_tasks
[PASS] write request -> pending_confirmation, no success claim
[PASS] cross-department request -> refused without tool call
smoke passed: contract demo only; this is not a model quality baseline
```

## Run locally

Start the fake agent:

```bash
python demo/fake_agent/app.py --port 18091
```

In another terminal:

```bash
python demo/smoke.py
```

## What this does not prove

- It does not measure model quality.
- It does not inspect real MySQL side effects.
- It does not replace the 100-case, 300-run baseline.
- It does not prove the production Java backend is bound to the evaluation database.
