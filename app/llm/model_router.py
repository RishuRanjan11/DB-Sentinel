from typing import Any
import re

from app.core.trace import logger
from app.llm.base import LLMProvider
from app.llm.complexity import (
    QueryComplexity,
    classify_question,
    required_capabilities,
)
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.model_registry import ModelConfig, ModelRegistry


class ModelRouter:
    """
    Registry-driven model selection.

    Routing policy:
      - SIMPLE  -> best available SIMPLE/Fast model.
      - COMPLEX -> best available COMPLEX/Pro model.
      - If no COMPLEX model is available, fall back to the best eligible
        lower-tier model according to registry priority/cost.
      - Provider names are never used to rank models.
      - A generation retry does not artificially skip models based on the
        retry number; provider/model availability is handled inside routing.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        providers: dict[str, LLMProvider],
    ):
        self.registry = registry
        self.providers = providers
        self.last_route_metadata: dict[str, Any] = {}

    def _build_route(
        self,
        complexity: str,
        capabilities: set[str],
    ) -> list[ModelConfig]:
        requested_tier = (
            QueryComplexity.SIMPLE.value
            if complexity == QueryComplexity.SIMPLE.value
            else QueryComplexity.COMPLEX.value
        )

        preferred = self.registry.find_models_by_tier(
            requested_tier,
            capabilities,
        )

        if preferred:
            return preferred

        # No model exists at the requested tier. Fall back to any
        # eligible model, ranked by registry priority/cost rather than
        # provider or model-name assumptions.
        fallback = self.registry.find_capable_models(
            capabilities
        )

        return fallback

    def generate(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: Any,
        complexity: str | None = None,
        classification_text: str | None = None,
        attempt_number: int | None = None,
    ):
        if complexity is None:
            complexity = classify_question(
                classification_text
                if classification_text is not None
                else user_prompt
            )

        attempt_number = max(
            1,
            int(attempt_number or 1),
        )

        capabilities = required_capabilities(
            complexity
        )

        ordered_models = self._build_route(
            complexity=complexity,
            capabilities=capabilities,
        )

        if not ordered_models:
            raise RuntimeError(
                "No available LLM models satisfy the required capabilities."
            )

        logger.debug(
            "Route requested_complexity=%s attempt=%s "
            "capabilities=%s candidates=%s",
            complexity,
            attempt_number,
            sorted(capabilities),
            [
                f"{model.provider}/{model.model}/{model.tier}"
                for model in ordered_models
            ],
        )

        last_error = None
        unavailable_models = []

        for position, model_config in enumerate(
            ordered_models,
            start=1,
        ):
            provider = self.providers.get(
                model_config.provider
            )

            if provider is None:
                unavailable_models.append(
                    f"{model_config.provider}/{model_config.model}: provider unavailable"
                )
                continue

            try:
                healthy = provider.health_check()
            except Exception as error:
                healthy = False
                logger.warning(
                    "Provider health check failed provider=%s error=%s",
                    model_config.provider,
                    error,
                )

            if not healthy:
                unavailable_models.append(
                    f"{model_config.provider}/{model_config.model}: health check failed"
                )
                continue

            degraded = (
                complexity == QueryComplexity.COMPLEX.value
                and model_config.tier
                != QueryComplexity.COMPLEX.value
            )

            logger.debug(
                "Trying model=%s/%s tier=%s priority=%s cost_rank=%s "
                "degraded=%s route_position=%s/%s",
                model_config.provider,
                model_config.model,
                model_config.tier,
                model_config.priority,
                model_config.cost_rank,
                degraded,
                position,
                len(ordered_models),
            )

            try:
                result = provider.generate(
                    model=model_config.model,
                    system_instruction=system_instruction,
                    user_prompt=user_prompt,
                    response_model=response_model,
                )

                self.last_route_metadata = {
                    "requested_complexity": complexity,
                    "actual_complexity": model_config.tier,
                    "degraded_mode": degraded,
                    "provider": model_config.provider,
                    "model": model_config.model,
                    "attempt_number": attempt_number,
                    "route_position": position,
                    "candidate_count": len(ordered_models),
                    "fallback_reason": (
                        "No model was available at the requested "
                        "complexity tier."
                        if degraded
                        else None
                    ),
                }

                return result

            except Exception as error:
                last_error = error

                logger.warning(
                    "Model failed provider=%s model=%s tier=%s error=%s",
                    model_config.provider,
                    model_config.model,
                    model_config.tier,
                    error,
                )

        details = (
            "; ".join(unavailable_models)
            if unavailable_models
            else "none"
        )

        raise RuntimeError(
            "All eligible LLM models failed. "
            f"Unavailable candidates: {details}"
        ) from last_error

def create_default_model_router() -> ModelRouter:
    registry = ModelRegistry()
    providers: dict[str, LLMProvider] = {}

    try:
        providers["gemini"] = GeminiProvider()
    except ValueError as error:
        logger.warning("Gemini provider unavailable: %s", error)

    try:
        providers["groq"] = GroqProvider()
    except ValueError as error:
        logger.warning("Groq provider unavailable: %s", error)

    for provider_name, provider in providers.items():
        try:
            discovered_models = provider.list_models()
            logger.debug(
                "Discovered %d usable models from %s",
                len(discovered_models),
                provider_name,
            )

            for discovered in discovered_models:
                capabilities = {"sql_generation"}
                if discovered.get("structured_output", False):
                    capabilities.add("structured_output")
                if discovered.get("reasoning", False):
                    capabilities.add("reasoning")

                tier = (
                    QueryComplexity.COMPLEX
                    if discovered.get("reasoning", False)
                    else QueryComplexity.SIMPLE
                )

                registry.register(
                    ModelConfig(
                        provider=provider_name,
                        model=discovered["model"],
                        tier=tier,
                        priority=discovered.get("priority", 1),
                        cost_rank=discovered.get("cost_rank", 1.0),
                        context_window=(
                            discovered.get("context_window")
                            or discovered.get("input_token_limit")
                        ),
                        structured_output=discovered.get("structured_output", False),
                        reasoning=discovered.get("reasoning", False),
                        capabilities=frozenset(capabilities),
                    )
                )
        except Exception as error:
            logger.warning("Model discovery failed provider=%s error=%s", provider_name, error)

    return ModelRouter(registry=registry, providers=providers)
