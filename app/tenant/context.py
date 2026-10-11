from dataclasses import dataclass, field


@dataclass(frozen=True)
class OrganizationContext:
    """Trusted request context for one SaaS organization."""

    organization_id: str
    user_id: str | None = None
    database_connection_ids: tuple[str, ...] = field(default_factory=tuple)

    def require_database(self, connection_id: str) -> None:
        if self.database_connection_ids and connection_id not in self.database_connection_ids:
            raise PermissionError(
                f"Database connection '{connection_id}' is not available to organization "
                f"'{self.organization_id}'."
            )
