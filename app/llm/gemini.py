import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.llm.base import LLMProvider
from app.core.trace import logger


load_dotenv()


class GeminiProvider(LLMProvider):

    STRUCTURED_OUTPUT_MODELS = {
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    }

    REASONING_MODELS = {"gemini-3.1-pro-preview"}

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

    @staticmethod
    def _build_response_schema(
        response_model: Any,
    ) -> dict:

        schema = response_model.model_json_schema()

        def clean(value):

            if isinstance(value, dict):

                cleaned = {}

                for key, item in value.items():

                    # Gemini's schema handling through the
                    # SDK can reject this Pydantic-generated
                    # field. The response model itself is
                    # still validated by Pydantic afterwards.
                    if key == "additionalProperties":
                        continue

                    cleaned[key] = clean(item)

                return cleaned

            if isinstance(value, list):
                return [
                    clean(item)
                    for item in value
                ]

            return value

        return clean(schema)

    def generate(
        self,
        model: str,
        system_instruction: str,
        user_prompt: str,
        response_model,
    ):

        response_schema = (
            self._build_response_schema(
                response_model
            )
        )

        logger.debug("Gemini request model=%s", model)

        response = self.client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        logger.debug("Gemini response model=%s", model)
        return response_model.model_validate_json(response.text)

    def list_models(self) -> list[dict]:

        response_models = self.client.models.list()

        models = []

        for model in response_models:

            name = getattr(
                model,
                "name",
                None,
            )

            if not name:
                continue

            model_id = name.replace(
                "models/",
                "",
            )

            if model_id not in self.STRUCTURED_OUTPUT_MODELS:
                continue

            models.append(
                {
                    "provider": "gemini",
                    "model": model_id,
                    "display_name": getattr(
                        model,
                        "display_name",
                        None,
                    ),
                    "structured_output": True,
                    "reasoning": (
                        model_id
                        in self.REASONING_MODELS
                    ),
                    "input_token_limit": getattr(
                        model,
                        "input_token_limit",
                        None,
                    ),
                    "output_token_limit": getattr(
                        model,
                        "output_token_limit",
                        None,
                    ),
                }
            )

        return models

    def health_check(self) -> bool:

        try:
            return len(self.list_models()) > 0

        except Exception:
            return False