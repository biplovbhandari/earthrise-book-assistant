from __future__ import annotations

from pathlib import Path

from earthrise_rag.models.scored_chunk import ScoredChunk

_PROMPT_PATH = Path(__file__).parent / "PROMPT.md"
SYSTEM_PROMPT = _PROMPT_PATH.read_text().strip()


class DefaultContextBuilder:
    """Assembles system + user messages from retrieved chunks."""

    def build(self, question: str, chunks: list[ScoredChunk]) -> list[dict[str, str]]:
        """Assemble system and user messages from retrieved chunks.

        Args:
            question: The user's natural language question.
            chunks: Ranked chunks from retrieval.

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

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
