import json
import os

from dotenv import load_dotenv
from groq import Groq

from app.llm.base import LLMProvider
from app.core.trace import logger

load_dotenv()


class GroqProvider(LLMProvider):
    STRUCTURED_OUTPUT_MODELS = {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    }

    REASONING_MODELS = {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    }

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from .env")
        self.client = Groq(api_key=api_key)

    @staticmethod
    def _extract_error_text(error: Exception) -> str:
        return str(error).lower()

    def _request(
        self,
        model: str,
        system_instruction: str,
        user_prompt: str,
        response_format: dict,
    ):
        return self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format=response_format,
        )

    def generate(
        self,
        model: str,
        system_instruction: str,
        user_prompt: str,
        response_model,
    ):
        schema = response_model.model_json_schema()

        structured_prompt = (
            user_prompt
            + "\n\nReturn ONLY valid JSON matching the requested schema. "
              "Do not include markdown fences or explanatory text."
        )

        logger.debug("Groq request model=%s", model)

        try:
            response = self._request(
                model=model,
                system_instruction=system_instruction,
                user_prompt=structured_prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except Exception as error:
            error_text = self._extract_error_text(error)

            # Some Groq models expose JSON mode but reject strict JSON
            # schemas for particular Pydantic constructs. Fall back to
            # generic JSON mode before declaring the model unusable.
            if "json_validate_failed" not in error_text and "failed to validate json" not in error_text:
                logger.warning("Groq request failed model=%s error=%s", model, error)
                raise

            logger.warning(
                "Groq JSON-schema mode rejected by model=%s; retrying with generic JSON mode",
                model,
            )
            response = self._request(
                model=model,
                system_instruction=system_instruction,
                user_prompt=(
                    structured_prompt
                    + "\nThe required JSON object shape is:\n"
                    + json.dumps(schema, ensure_ascii=False)
                ),
                response_format={"type": "json_object"},
            )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty response.")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            logger.warning("Groq returned invalid JSON model=%s", model)
            raise ValueError("Groq returned invalid JSON.") from error

        return response_model.model_validate(payload)

    def list_models(self) -> list[dict]:
        response = self.client.models.list()
        models = []
        for model in response.data:
            model_id = model.id
            if model_id not in self.STRUCTURED_OUTPUT_MODELS:
                continue
            models.append(
                {
                    "provider": "groq",
                    "model": model_id,
                    "display_name": model_id,
                    "active": getattr(model, "active", True),
                    "context_window": getattr(model, "context_window", None),
                    "structured_output": True,
                    "reasoning": model_id in self.REASONING_MODELS,
                }
            )
        return models

    def health_check(self) -> bool:
        try:
            return len(self.list_models()) > 0
        except Exception:
            return False
