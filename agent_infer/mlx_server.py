"""OpenAI-compatible HTTP server backed by the MLX engine.

Start directly::

    python -m agent_infer.mlx_server \\
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \\
        --host 127.0.0.1 --port 8080

Endpoints
---------
GET  /health
GET  /v1/models
POST /v1/completions       (OpenAI text completion, streaming supported)
POST /v1/chat/completions  (OpenAI chat completion, streaming supported)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _sampling_from_body(body: dict) -> "SamplingParams":  # noqa: F821
    from agent_infer.mlx_backend import SamplingParams
    return SamplingParams(
        max_tokens=body.get("max_tokens") or 256,
        temperature=body.get("temperature", 0.0),
        top_p=body.get("top_p", 1.0),
        top_k=body.get("top_k", 0),
        repetition_penalty=body.get("repetition_penalty", 1.0),
        stop=body.get("stop") or [],
        seed=body.get("seed"),
    )


def _chatml_from_messages(messages: list[dict], tools: list[dict] | None) -> str:
    """Minimal ChatML formatter for /v1/chat/completions."""
    parts: list[str] = []

    # Inject tool definitions into the system turn.
    if tools:
        tool_json = json.dumps(tools, ensure_ascii=False, indent=2)
        sys_content = (
            "You are a helpful assistant.\n\n"
            "Available tools (call them as JSON inside <tool_call>...</tool_call>):\n"
            + tool_json
        )
        parts.append(f"<|im_start|>system\n{sys_content}<|im_end|>")
        # Skip any existing system message so we don't duplicate it.
        messages = [m for m in messages if m.get("role") != "system"]

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ASGI app builder
# ---------------------------------------------------------------------------

def build_app(engine: "MLXEngine"):  # noqa: F821
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route
    except ImportError as exc:
        raise RuntimeError(
            "HTTP server requires:  pip install starlette uvicorn"
        ) from exc

    model_name = engine.model_id

    # ------------------------------------------------------------------ /health
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "model": model_name})

    # ------------------------------------------------------------------ /v1/models
    async def list_models(_: Request) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx",
            }],
        })

    # ------------------------------------------------------------------ /v1/completions
    async def completions(request: Request) -> Any:
        body = await request.json()
        prompt: str = body.get("prompt", "")
        stream: bool = body.get("stream", False)
        params = _sampling_from_body(body)
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            async def sse_stream():
                async for text, is_final, usage in engine.generate_stream(prompt, params):
                    finish = (usage or {}).get("finish_reason") if is_final else None
                    chunk: dict[str, Any] = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "text": text,
                            "index": 0,
                            "finish_reason": finish,
                        }],
                    }
                    if is_final and usage:
                        chunk["usage"] = {
                            "prompt_tokens": usage["prompt_tokens"],
                            "completion_tokens": usage["completion_tokens"],
                            "total_tokens": usage["total_tokens"],
                        }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_stream(), media_type="text/event-stream")

        # Non-streaming: run in executor so we don't block the event loop.
        result = await asyncio.get_event_loop().run_in_executor(
            None, engine.generate, prompt, params
        )
        return JSONResponse({
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "text": result.text,
                "index": 0,
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        })

    # ------------------------------------------------------------------ /v1/chat/completions
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        messages: list[dict] = body.get("messages", [])
        tools: list[dict] | None = body.get("tools")
        stream: bool = body.get("stream", False)
        params = _sampling_from_body(body)
        prompt = _chatml_from_messages(messages, tools)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if stream:
            async def sse_stream():
                # Send an initial empty delta with role set.
                opening: dict[str, Any] = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "delta": {"role": "assistant", "content": ""},
                        "index": 0,
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(opening)}\n\n"

                async for text, is_final, usage in engine.generate_stream(prompt, params):
                    finish = (usage or {}).get("finish_reason") if is_final else None
                    chunk: dict[str, Any] = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "delta": {"content": text},
                            "index": 0,
                            "finish_reason": finish,
                        }],
                    }
                    if is_final and usage:
                        chunk["usage"] = {
                            "prompt_tokens": usage["prompt_tokens"],
                            "completion_tokens": usage["completion_tokens"],
                            "total_tokens": usage["total_tokens"],
                        }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_stream(), media_type="text/event-stream")

        result = await asyncio.get_event_loop().run_in_executor(
            None, engine.generate, prompt, params
        )
        return JSONResponse({
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "message": {"role": "assistant", "content": result.text},
                "index": 0,
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        })

    app = Starlette(routes=[
        Route("/health", health),
        Route("/v1/models", list_models),
        Route("/v1/completions", completions, methods=["POST"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ])
    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX inference HTTP server (OpenAI-compatible)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Local model path or HuggingFace repo "
             "(e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--max-memory-gb", type=float, default=None,
        help="Hard cap on model memory in GB (default: 60%% of available)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from agent_infer.mlx_backend import MLXEngine
    from agent_infer.resource_guard import check_apple_silicon, check_memory

    # Pre-flight checks before loading anything heavy.
    check_apple_silicon()
    check_memory()

    engine = MLXEngine(args.model)
    engine.load(max_memory_gb=args.max_memory_gb)

    app = build_app(engine)

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "HTTP server requires:  pip install uvicorn"
        ) from exc

    print(f"\nMLX server ready  →  http://{args.host}:{args.port}")
    print(f"  Model   : {args.model}")
    print(f"  Routes  : /health  /v1/models  /v1/completions  /v1/chat/completions\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
