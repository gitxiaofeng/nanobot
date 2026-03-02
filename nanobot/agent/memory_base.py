"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from nanobot.agent.memory import MemoryStore
from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session

"""
`memory` 的“读取逻辑”分两条线：`长期记忆文件` 和 `会话历史`，最终在每轮请求拼成上下文。

1. **启动时确保 memory 文件存在**
`sync_workspace_templates` 会创建 `memory/MEMORY.md`、`memory/HISTORY.md`（若不存在）。
见 [commands.py:188](/Users/weiyan/Documents/code/nanobot/nanobot/cli/commands.py:188), [commands.py:266](/Users/weiyan/Documents/code/nanobot/nanobot/cli/commands.py:266), [helpers.py:59](/Users/weiyan/Documents/code/nanobot/nanobot/utils/helpers.py:59)

2. **每次来消息先读 session，再取“未归档历史”**
`AgentLoop` 里先 `get_or_create`，然后 `session.get_history(max_messages=memory_window)`。
见 [loop.py:362](/Users/weiyan/Documents/code/nanobot/nanobot/agent/loop.py:362), [loop.py:421](/Users/weiyan/Documents/code/nanobot/nanobot/agent/loop.py:421)

3. **`get_history` 的读取规则**
只取 `messages[last_consolidated:]`（即未归档部分），再截断最近 `max_messages`，并且裁掉开头非 user 消息避免 tool orphan。
见 [manager.py:45](/Users/weiyan/Documents/code/nanobot/nanobot/session/manager.py:45), [manager.py:47](/Users/weiyan/Documents/code/nanobot/nanobot/session/manager.py:47), [manager.py:50](/Users/weiyan/Documents/code/nanobot/nanobot/session/manager.py:50)

4. **系统提示词里读取长期记忆**
`ContextBuilder.build_system_prompt()` 调 `MemoryStore.get_memory_context()`，后者实际 `read_long_term()` 读取 `memory/MEMORY.md`。  
注意：`HISTORY.md` 不会自动注入上下文，只是归档日志。
见 [context.py:34](/Users/weiyan/Documents/code/nanobot/nanobot/agent/context.py:34), [memory.py:53](/Users/weiyan/Documents/code/nanobot/nanobot/agent/memory.py:53), [memory.py:65](/Users/weiyan/Documents/code/nanobot/nanobot/agent/memory.py:65)

5. **触发归档后，后续“读取窗口”会前移**
当 `unconsolidated >= memory_window` 时异步归档；归档成功会更新 `session.last_consolidated`。之后 `get_history` 只读更新后的未归档尾部。
见 [loop.py:398](/Users/weiyan/Documents/code/nanobot/nanobot/agent/loop.py:398), [loop.py:406](/Users/weiyan/Documents/code/nanobot/nanobot/agent/loop.py:406), [memory.py:145](/Users/weiyan/Documents/code/nanobot/nanobot/agent/memory.py:145)

6. **`/new` 的特殊读取路径**
`/new` 会拿 `session.messages[session.last_consolidated:]` 做 `archive_all=True` 归档，再清空 session。
见 [loop.py:371](/Users/weiyan/Documents/code/nanobot/nanobot/agent/loop.py:371), [loop.py:375](/Users/weiyan/Documents/code/nanobot/nanobot/agent/loop.py:375), [loop.py:389](/Users/weiyan/Documents/code/nanobot/nanobot/agent/loop.py:389)

核心结论：  
- **真正每轮自动读取进模型的是 `MEMORY.md` + `未归档 session 历史`**。  
- **`HISTORY.md` 是可检索归档，不自动进 prompt**。
"""

_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStoreBase(MemoryStore):
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

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
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning("Memory consolidation: unexpected arguments type {}", type(args).__name__)
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False
