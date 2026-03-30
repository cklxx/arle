"""Shell command execution tool."""

from __future__ import annotations

import asyncio
import json

from agent_infer.tools.base import BaseTool


class ShellExecTool(BaseTool):
    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Execute a shell command and return stdout/stderr. Use for system operations, file management, or running programs."

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
            },
            "required": ["command"],
        }

    async def execute(self, *, command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        except asyncio.TimeoutError:
            proc.kill()
            return json.dumps({"error": "Execution timed out (60s)"})

        result = {}
        if stdout:
            result["stdout"] = stdout.decode(errors="replace")
        if stderr:
            result["stderr"] = stderr.decode(errors="replace")
        result["returncode"] = proc.returncode
        return json.dumps(result)
