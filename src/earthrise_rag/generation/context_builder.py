from __future__ import annotations

from pathlib import Path

from earthrise_rag.models.scored_chunk import ScoredChunk

_PROMPT_PATH = Path(__file__).parent / "PROMPT.md"
SYSTEM_PROMPT = _PROMPT_PATH.read_text().strip()

_MAX_HISTORY_CHARS = 4000


class DefaultContextBuilder:
    """Assembles system + user messages from retrieved chunks."""

    @staticmethod
    def _xml_escape(text: str) -> str:
        """Escape XML-special characters to prevent tag injection."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _format_history(history: list[dict[str, str]]) -> str:
        """Format history as XML-delimited turns, capped by total char budget.

        Content is XML-escaped to prevent injection of closing tags.
        Escaping happens before truncation so the cap applies to final output size.
        """
        _PRE_ESCAPE_CAP = 2000
        _POST_ESCAPE_CAP = 800

        turns = []
        for msg in history:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            content = str(msg.get("content", ""))
            if len(content) > _PRE_ESCAPE_CAP:
                content = content[:_PRE_ESCAPE_CAP]
            content = DefaultContextBuilder._xml_escape(content)
            if len(content) > _POST_ESCAPE_CAP:
                content = content[:_POST_ESCAPE_CAP] + "..."
            turns.append(f'<turn role="{role}">{content}</turn>')

        result = "\n".join(turns)
        while len(result) > _MAX_HISTORY_CHARS and turns:
            turns.pop(0)
            result = "\n".join(turns)
        return result

    def build(
        self,
        question: str,
        chunks: list[ScoredChunk],
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble system and user messages from retrieved chunks.

        Args:
            question: The user's natural language question.
            chunks: Ranked chunks from retrieval.
            history: Optional prior conversation turns for reference resolution.

        Returns:
            List of message dicts (system + user) ready for LLMClient.chat().
        """
        context_parts = []
        for i, scored in enumerate(chunks, 1):
            source = scored.chunk.metadata.get("source_path", "unknown")
            section = scored.chunk.metadata.get("section", "")
            header = f"[{i}] {source}"
            if section:
                header += f" - {section}"
            context_parts.append(f"{header}\n{scored.chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            history_block = self._format_history(history)
            user_content = (
                f"<history>\n{history_block}\n</history>\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
        else:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"

        messages.append({"role": "user", "content": user_content})
        return messages
