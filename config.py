import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    # Paths
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    KNOWLEDGE_BASE_PATH: str = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge/data")

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Agent settings
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))


config = Config()
