import os

from dotenv import load_dotenv


load_dotenv()


class Settings:

    APP_NAME: str = "DB-Sentinel"
    APP_VERSION: str = "0.1.0"

    ENVIRONMENT: str = os.getenv(
        "ENVIRONMENT",
        "development",
    )

    DEBUG: bool = os.getenv(
        "DEBUG",
        "false",
    ).lower() == "true"

    GEMINI_API_KEY: str | None = os.getenv(
        "GEMINI_API_KEY"
    )


settings = Settings()