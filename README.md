# agent-infer

Agent system with multi-turn conversation and tool calling, built on [Dynamo](https://github.com/cklxx/dynamo) (inference orchestration) + [Pegainfer](https://github.com/cklxx/pegainfer) (Rust+CUDA inference engine).

## Architecture

```
User Request
     ↓
Agent Loop (agent_infer/)
  ├─ Chat formatter (Qwen3 ChatML)
  ├─ Tool call parser
  └─ Tool executor (python, shell, file)
     ↓
LLM Client
  ├─ Direct: pegainfer /v1/completions
  └─ Orchestrated: Dynamo frontend /v1/chat/completions
     ↓
Pegainfer (Rust+CUDA inference)
  └─ Qwen3-4B / Qwen3-8B / Qwen3.5-4B
```

## Quick Start

### 1. Start Pegainfer

```bash
./scripts/start_pegainfer.sh pegainfer/models/Qwen3-4B
```

### 2. Run Agent (Direct Mode)

```bash
pip install -e .
python -m agent_infer --url http://localhost:8000
```

### 3. Run Agent (via Dynamo)

```bash
# Terminal 1: Start Dynamo + pegainfer backend
./scripts/start_dynamo.sh

# Terminal 2: Start agent
./scripts/start_agent.sh dynamo
```

## Usage

### Interactive REPL

```bash
python -m agent_infer
```

### Single Query

```bash
python -m agent_infer query "What is 2^100?"
```

### HTTP Server

```bash
python -m agent_infer serve --port 9000
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `python` | Execute Python code |
| `shell` | Execute shell commands |
| `file` | Read, write, list files |

## Project Structure

```
agent-infer/
├── agent_infer/              # Agent framework (Python)
│   ├── agent.py              # Agent loop: generate → parse → execute → repeat
│   ├── chat.py               # Message types, ChatML formatter, tool call parser
│   ├── client.py             # Async LLM client (pegainfer / OpenAI-compatible)
│   └── tools/                # Tool registry and built-in tools
├── dynamo/                   # Dynamo source (inference orchestration)
│   └── components/src/dynamo/pegainfer/  # Pegainfer backend for Dynamo
├── pegainfer/                # Pegainfer source (Rust+CUDA inference engine)
└── scripts/                  # Startup scripts
```
