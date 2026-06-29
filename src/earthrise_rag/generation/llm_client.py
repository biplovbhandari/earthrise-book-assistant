from __future__ import annotations


class OpenAICompatibleClient:
    """LLM client using the OpenAI-compatible chat completions API."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        """Initialize the client with connection parameters.

        Args:
            base_url: API endpoint URL (Ollama, NASA proxy, etc.).
            api_key: Authentication key for the API.
            model: Model name to use for completions.
            timeout: Request timeout in seconds.
        """
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the generated response.

        Returns:
            The generated text content.

        Raises:
            RuntimeError: If the response has no choices or empty content.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response.choices:
            raise RuntimeError("LLM returned no choices")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("LLM returned empty content")
        return content
