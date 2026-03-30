"""Python code execution tool."""

from __future__ import annotations

import asyncio
import json

from agent_infer.tools.base import BaseTool


class PythonExecTool(BaseTool):
    @property
    def name(self) -> str:
        return "python"

    @property
    def description(self) -> str:
        return "Execute Python code and return the output. Use for calculations, data processing, or any programmatic task."

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
            },
            "required": ["code"],
        }

    async def execute(self, *, code: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except asyncio.TimeoutError:
            proc.kill()
            return json.dumps({"error": "Execution timed out (30s)"})

        result = {}
        if stdout:
            result["stdout"] = stdout.decode(errors="replace")
        if stderr:
            result["stderr"] = stderr.decode(errors="replace")
        result["returncode"] = proc.returncode
        return json.dumps(result)
