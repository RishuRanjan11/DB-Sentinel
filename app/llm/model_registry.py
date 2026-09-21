from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    tier: str

    priority: int = 1

    # Lower value = cheaper model.
    # This is intentionally configurable instead of hardcoded
    # into the router.
    cost_rank: float = 1.0

    context_window: int | None = None

    structured_output: bool = False
    reasoning: bool = False

    capabilities: frozenset[str] = field(
        default_factory=frozenset
    )

    available: bool = True


class ModelRegistry:

    def __init__(self):
        self._models: list[ModelConfig] = []

    def register(
        self,
        model: ModelConfig,
    ) -> None:
        self._models.append(model)

    def clear(self) -> None:
        self._models.clear()

    def get_all_models(
        self,
    ) -> list[ModelConfig]:
        return list(self._models)

    def find_capable_models(
        self,
        required_capabilities: set[str],
    ) -> list[ModelConfig]:

        results = []

        for model in self._models:

            if not model.available:
                continue

            if not required_capabilities.issubset(
                model.capabilities
            ):
                continue

            results.append(model)

        return sorted(
            results,
            key=lambda model: (
                model.priority,
                model.cost_rank,
            ),
        )

    def find_models_by_tier(
        self,
        tier: str,
        required_capabilities: set[str],
    ) -> list[ModelConfig]:

        return sorted(
            [
                model
                for model in self._models
                if (
                    model.available
                    and model.tier == tier
                    and required_capabilities.issubset(
                        model.capabilities
                    )
                )
            ],
            key=lambda model: (
                model.priority,
                model.cost_rank,
            ),
        )

    def find_cheapest_models(
        self,
        tier: str,
        required_capabilities: set[str],
    ) -> list[ModelConfig]:

        return sorted(
            [
                model
                for model in self._models
                if (
                    model.available
                    and model.tier == tier
                    and required_capabilities.issubset(
                        model.capabilities
                    )
                )
            ],
            key=lambda model: (
                model.cost_rank,
                model.priority,
            ),
        )