"""
Gemini client for AutoPR.

Uses Google's official google-genai SDK and Gemini 3.5 Flash-Lite.
"""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types


class GeminiClient:
    """
    Small wrapper around the Gemini API.

    AutoPRAgent only needs one method:

        generate(system_prompt, history) -> str
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your local .env file."
            )

        self.model = (
            model
            or os.getenv(
                "GEMINI_MODEL",
                self.DEFAULT_MODEL,
            )
        )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
    ) -> str:
        """
        Send the AutoPR conversation to Gemini.

        Gemini is instructed to return JSON only.
        """

        if not system_prompt:
            raise ValueError(
                "system_prompt cannot be empty."
            )

        if not history:
            raise ValueError(
                "history cannot be empty."
            )

        conversation_parts: list[str] = []

        for message in history:
            role = str(
                message.get("role", "user")
            ).upper()

            content = str(
                message.get("content", "")
            )

            if not content:
                continue

            conversation_parts.append(
                f"{role}:\n{content}"
            )

        conversation = "\n\n".join(
            conversation_parts
        )

        if not conversation:
            raise ValueError(
                "Conversation history contains no content."
            )

        try:
            response = (
                self.client.models.generate_content(
                    model=self.model,
                    contents=conversation,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.2,
                        max_output_tokens=2000,
                    ),
                )
            )

        except Exception as exc:
            raise RuntimeError(
                f"Gemini API request failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        text = getattr(
            response,
            "text",
            None,
        )

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return text.strip()

    def test_connection(self) -> str:
        """
        Simple connection test.

        Useful before starting the full AutoPR workflow.
        """

        try:
            response = (
                self.client.models.generate_content(
                    model=self.model,
                    contents=(
                        "Reply with exactly this word: "
                        "CONNECTED"
                    ),
                    config=types.GenerateContentConfig(
                        max_output_tokens=20,
                    ),
                )
            )

        except Exception as exc:
            raise RuntimeError(
                f"Gemini connection test failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        text = getattr(
            response,
            "text",
            None,
        )

        if not text:
            raise RuntimeError(
                "Gemini connection test returned "
                "an empty response."
            )

        return text.strip()
