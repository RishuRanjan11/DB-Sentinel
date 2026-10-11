from app.agent.investigation import InvestigationEngine
from app.conversation.session import ConversationSession
from app.tenant.context import OrganizationContext
from app.database.data_dictionary import DatabaseDictionary


class ConversationalAnalyst:
    """Thin conversation layer around the existing investigation engine.

    The investigation engine remains the source of truth for security,
    verification and database analysis. This layer only owns conversation
    continuity and tenant-scoped database selection.
    """

    def __init__(self, engine: InvestigationEngine):
        self.engine = engine

    def ask(
        self,
        session: ConversationSession,
        question: str,
        connection_id: str,
        data_dictionary: DatabaseDictionary | None = None,
    ) -> dict:
        context = OrganizationContext(
            organization_id=session.organization_id,
            database_connection_ids=session.database_connection_ids,
        )
        context.require_database(connection_id)

        result = self.engine.investigate(
            connection_id=connection_id,
            question=question,
            organization_id=session.organization_id,
            conversation_history=session.history(),
            data_dictionary=data_dictionary,
        )
        session.add_turn(question, result)
        return result
