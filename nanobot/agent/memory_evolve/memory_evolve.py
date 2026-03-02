"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir
from nanobot.agent.memory import MemoryStore

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session

"""
工程里有两套 memory 读取逻辑，建议区分看：

1) Agent“记忆库”读取（文件级）

会话启动时注入记忆：get_system_prompt() 调用 get_global_memory()，读取 memory/global_mem_insight.txt 和 assets/insight_fixed_structure.txt，拼到系统提示词里。见 agentmain.py (line 16)、ga.py (line 488)。
运行中周期注入：每 10 轮在 next_prompt_patcher() 再次调用 get_global_memory()。见 ga.py (line 480)。
工具主动读取：模型调用 file_read 时走 do_file_read()，底层是通用 file_read()（支持 start/count/keyword）。见 ga.py (line 380)、ga.py (line 205)。
读取统计：do_file_read() 会调用 log_memory_access(path)，只要路径里含 memory 就累计到 memory/file_access_stats.json。见 ga.py (line 151)、ga.py (line 396)。
读取记忆/SOP后会加提示：若路径含 memory 或 sop，会在下一轮提示“提取关键点更新 working memory”。见 ga.py (line 397)。
结算记忆时会读 L0 SOP：start_long_term_update 直接读 ./memory/memory_management_sop.md。见 ga.py (line 454)。
2) 进程“内存扫描”读取（ReadProcessMemory）

独立在 memory/mem_scanner.py，不是上面那套记忆库。
流程：OpenProcess -> VirtualQueryEx 枚举内存区 -> 对可读区 ReadProcessMemory -> 用 YARA 匹配模式 -> 返回命中上下文。见 mem_scanner.py (line 63)。
这是 Windows 进程内存取证/扫描逻辑，和 global_mem*.txt 的长期记忆管理是两条链路。
如果你想，我可以下一步给你画“从一次用户请求到 memory 被读取”的完整时序图（按函数调用顺序）。

基本不“自动协同”，而是并行存在、由 Agent 在任务中手动桥接：

记忆库读取链路（L0/L1/L2/L3，文件）
负责给模型提供长期知识与规则（get_global_memory 注入 + file_read 读 memory/*.md|txt）。
输出是提示词上下文与文本内容，指导“该做什么”。
进程内存读取链路（mem_scanner.py）
负责从目标进程 ReadProcessMemory 拿原始字节并匹配模式。
输出是命中地址/hex/ascii，解决“目标程序里实际有什么”。
协同方式（当前实现）
不是代码层直接互调；ga.py 没有直接调用 scan_memory()。
协同靠 Agent 的工具编排：先从记忆库/SOP 读方法（file_read memory/*）-> 再 code_run 执行 mem_scanner.py 或导入 scan_memory -> 得到结果后再按 memory_management_sop 写回长期记忆（file_patch 到 L2/L3，必要时更新 L1）。
一句话：记忆库提供策略，进程内存提供数据，Agent 负责在两者间搬运与沉淀。
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


class MemoryStoreEvolve(MemoryStore):
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
