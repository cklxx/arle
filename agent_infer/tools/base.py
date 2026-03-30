"""Base class for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def parameters_schema(self) -> dict:
        """Return JSON Schema for tool parameters."""

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool and return result as string."""

    def openai_schema(self) -> dict:
        """Return OpenAI function tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }
