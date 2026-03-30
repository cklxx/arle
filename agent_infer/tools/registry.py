"""Tool registry for managing and executing tools."""

from __future__ import annotations

import json
import logging

from agent_infer.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def has_tools(self) -> bool:
        return len(self._tools) > 0

    def list_schemas(self) -> list[dict]:
        """Return all tool schemas in OpenAI function-calling format."""
        return [tool.openai_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = await tool.execute(**arguments)
            return result
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": str(e)})
