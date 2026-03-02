"""Memory system for persistent agent memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from loguru import logger

MemoryBackend = Literal["base", "evolve"]

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


class MemoryStore(ABC):
    @abstractmethod
    def read_long_term(self) -> str:
        pass
    
    @abstractmethod
    def write_long_term(self, content: str) -> None:
        pass
    
    @abstractmethod
    def append_history(self, entry: str) -> None:
        pass

    @abstractmethod
    def get_memory_context(self) -> str:
        pass
    
    @abstractmethod
    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        pass
        
def create_memory_store(workspace: Path, backend: MemoryBackend = "base") -> MemoryStore:
    if backend == "base":
        from nanobot.agent.memory_base import MemoryStoreBase
        return MemoryStoreBase(workspace)
    elif backend == "evolve":
        from nanobot.agent.memory_evolve.memory_evolve import MemoryStoreEvolve
        return MemoryStoreEvolve(workspace)
    else:
        raise ValueError(f"Unknown memory backend: {backend}")