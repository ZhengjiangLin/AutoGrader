import time
from typing import Any

import requests

from .config import Settings


class LLMClient:
    """Client responsible for communicating with any LLM (OpenAI, Azure, or compatible)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        """Build HTTP headers for the LLM API request."""
        if not self.settings.llm_api_key:
            raise ValueError("Missing LLM_API_KEY. Please set it in your .env file.")

        if self.settings.is_azure_openai:
            # Azure OpenAI uses api-key header
            return {
                "api-key": self.settings.llm_api_key,
                "Content-Type": "application/json",
            }
        # OpenAI and most compatible providers use Bearer token
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    def _request_url(self, model: str) -> str:
        """Build the correct API endpoint URL."""
        if self.settings.is_azure_openai:
            endpoint = self.settings.azure_openai_endpoint.rstrip("/")
            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is required when using Azure.")
            api_version = self.settings.azure_openai_api_version
            return (
                f"{endpoint}/openai/deployments/{model}/chat/completions"
                f"?api-version={api_version}"
            )
        # Normal OpenAI-style endpoint
        return self.settings.resolved_llm_api_url

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4000,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send a chat request to the LLM and return the raw API response."""
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not self.settings.is_azure_openai:
            payload["model"] = model
        if response_format:
            payload["response_format"] = response_format

        max_attempts = max(1, self.settings.llm_max_retries + 1)
        request_url = self._request_url(model)

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    request_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.request_timeout,
                )

                # Only retry on temporary server errors
                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                retryable = status_code in {429, 500, 502, 503, 504}
                if not retryable or attempt >= max_attempts:
                    raise
            except (requests.Timeout, requests.ConnectionError):
                if attempt >= max_attempts:
                    raise

            # Wait before retry
            sleep_seconds = self.settings.llm_retry_backoff_seconds * attempt
            time.sleep(max(0.0, sleep_seconds))

        raise RuntimeError("LLM request failed after maximum retries.")

    @staticmethod
    def message_text(api_result: dict[str, Any]) -> str:
        """Extract the actual text content from the LLM API response."""
        choices = api_result.get("choices", [])
        if not choices:
            raise ValueError(f"Model response has no choices: {api_result}")

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            # Some vision models return content as list of parts
            parts = [item.get("text", "") for item in content if item.get("type") == "text"]
            return "\n".join(parts).strip()

        raise ValueError(f"Could not parse model output content: {content}")