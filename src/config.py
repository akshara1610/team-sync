from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "TeamSync"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"

    # LiveKit
    LIVEKIT_URL: str
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str

    # LLM Provider Configuration
    LLM_PROVIDER: str = "gemini"  # Options: "openai", "gemini", "anthropic"

    # OpenAI
    OPENAI_API_KEY: str = ""

    # Google Gemini
    GOOGLE_API_KEY: str = ""

    # Anthropic
    ANTHROPIC_API_KEY: str = ""

    # HuggingFace
    HF_TOKEN: str

    # JIRA
    JIRA_URL: str
    JIRA_EMAIL: str
    JIRA_API_TOKEN: str
    JIRA_PROJECT_KEY: str

    # Email Configuration (TeamSync bot account)
    TEAMSYNC_EMAIL: str = "teamsync990@gmail.com"  # Dedicated TeamSync email
    EMAIL_PASSWORD: str = ""  # Gmail App Password for teamsync990@gmail.com

    # Google Calendar
    GOOGLE_CREDENTIALS_FILE: str = "credentials.json"
    GOOGLE_TOKEN_FILE: str = "token.json"

    # Vector Store (ChromaDB)
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "meeting_transcripts"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Agent Configuration
    MAX_REFLECTION_ITERATIONS: int = 3
    SUMMARIZER_MODEL: str = "gemini-2.0-flash-exp"  # Gemini 2.0 Flash (latest, fastest)
    # Other options: "gemini-1.5-pro", "gemini-1.5-flash", "gpt-4-turbo-preview", "claude-3-5-sonnet-20241022"
    KNOWLEDGE_AGENT_TOP_K: int = 5

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
