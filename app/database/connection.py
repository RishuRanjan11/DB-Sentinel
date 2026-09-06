import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()


def create_db_engine() -> Engine:
    host = os.getenv("FLYBASE_DB_HOST")
    port = os.getenv("FLYBASE_DB_PORT", "5432")
    database = os.getenv("FLYBASE_DB_NAME")
    user = os.getenv("FLYBASE_DB_USER")

    if not all([host, database, user]):
        raise ValueError("Missing FlyBase database configuration in .env")

    database_url = (
        f"postgresql+psycopg2://"
        f"{user}@{host}:{port}/{database}"
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )


engine = create_db_engine()


def test_connection() -> bool:
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database();")
            )
            database_name = result.scalar()

            print(f"Connected to database: {database_name}")
            return True

    except Exception as error:
        print(f"Database connection failed: {error}")
        return False


if __name__ == "__main__":
    test_connection()