"""CLI entry point for agent-infer.

Modes:
- Interactive REPL: python -m agent_infer
- HTTP server: python -m agent_infer serve
- Single query: python -m agent_infer query "your question"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from agent_infer.agent import AgentConfig, AgentLoop
from agent_infer.client import ClientConfig, LLMClient
from agent_infer.tools import ToolRegistry
from agent_infer.tools.file_ops import FileOpsTool
from agent_infer.tools.python_exec import PythonExecTool
from agent_infer.tools.shell_exec import ShellExecTool


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(PythonExecTool())
    registry.register(ShellExecTool())
    registry.register(FileOpsTool())
    return registry


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent-infer",
        description="Agent system powered by Dynamo + Pegainfer",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="LLM server URL (pegainfer or dynamo frontend)",
    )
    parser.add_argument(
        "--mode",
        choices=["completions", "chat"],
        default="completions",
        help="Client mode: 'completions' for direct pegainfer, 'chat' for dynamo/openai",
    )
    parser.add_argument("--model", default="Qwen3-4B", help="Model name")
    parser.add_argument("--max-turns", type=int, default=10, help="Max tool-calling turns")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument("--no-tools", action="store_true", help="Disable built-in tools")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command")

    # Single query
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", help="Question to ask")

    # HTTP server
    serve_parser = subparsers.add_parser("serve", help="Start HTTP API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=9000)

    return parser.parse_args()


async def run_repl(agent: AgentLoop):
    """Interactive REPL mode."""
    print("agent-infer interactive mode")
    print(f"  Backend: {agent.config.client.base_url}")
    print(f"  Mode: {agent.config.client.mode}")
    print(f"  Model: {agent.config.client.model}")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        state = await agent.run(user_input)
        print(f"\n{state.last_response}\n")
        if state.turn_count > 0:
            print(f"  [{state.turn_count} tool-calling turn(s)]\n")


async def run_query(agent: AgentLoop, question: str):
    """Single query mode."""
    state = await agent.run(question)
    print(state.last_response)


async def run_serve(agent: AgentLoop, host: str, port: int):
    """HTTP API server mode."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        import uvicorn
    except ImportError:
        print("HTTP server requires: pip install starlette uvicorn")
        sys.exit(1)

    async def handle_chat(request: Request) -> JSONResponse:
        body = await request.json()
        messages = body.get("messages", [])
        if not messages:
            return JSONResponse({"error": "No messages provided"}, status_code=400)

        user_msg = messages[-1].get("content", "")
        state = await agent.run(user_msg)

        return JSONResponse({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": state.last_response,
                },
                "finish_reason": "stop" if state.status == "complete" else state.status,
            }],
            "usage": {"tool_turns": state.turn_count},
        })

    async def handle_health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    app = Starlette(routes=[
        Route("/v1/chat/completions", handle_chat, methods=["POST"]),
        Route("/health", handle_health),
    ])

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def main():
    args = parse_cli_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client_config = ClientConfig(
        base_url=args.url,
        mode=args.mode,
        model=args.model,
    )
    agent_config = AgentConfig(
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        client=client_config,
    )

    registry = ToolRegistry() if args.no_tools else build_registry()
    client = LLMClient(client_config)
    agent = AgentLoop(config=agent_config, registry=registry, client=client)

    if args.command == "query":
        asyncio.run(run_query(agent, args.question))
    elif args.command == "serve":
        asyncio.run(run_serve(agent, args.host, args.port))
    else:
        asyncio.run(run_repl(agent))


if __name__ == "__main__":
    main()
