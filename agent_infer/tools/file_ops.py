"""File operations tool."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent_infer.tools.base import BaseTool


class FileOpsTool(BaseTool):
    def __init__(self, working_dir: str = "."):
        self._working_dir = Path(working_dir).resolve()

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return "Read, write, or list files. Supports operations: read, write, list."

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list"],
                    "description": "File operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path (relative to working directory)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (only for write operation)",
                },
            },
            "required": ["operation", "path"],
        }

    def _resolve(self, path: str) -> Path:
        resolved = (self._working_dir / path).resolve()
        if not str(resolved).startswith(str(self._working_dir)):
            raise ValueError(f"Path escapes working directory: {path}")
        return resolved

    async def execute(self, *, operation: str, path: str, content: str = "") -> str:
        try:
            target = self._resolve(path)

            if operation == "read":
                if not target.exists():
                    return json.dumps({"error": f"File not found: {path}"})
                text = target.read_text(encoding="utf-8", errors="replace")
                # Truncate very large files
                if len(text) > 50000:
                    text = text[:50000] + "\n... (truncated)"
                return json.dumps({"content": text})

            elif operation == "write":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return json.dumps({"status": "ok", "path": str(target)})

            elif operation == "list":
                if not target.exists():
                    return json.dumps({"error": f"Directory not found: {path}"})
                entries = []
                for entry in sorted(target.iterdir()):
                    entries.append({
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else None,
                    })
                return json.dumps({"entries": entries[:200]})

            else:
                return json.dumps({"error": f"Unknown operation: {operation}"})

        except ValueError as e:
            return json.dumps({"error": str(e)})
        except OSError as e:
            return json.dumps({"error": str(e)})
