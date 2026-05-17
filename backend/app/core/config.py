"""Single source of settings. All values come from ``.env`` via
pydantic-settings — no secrets in code. See docs/architecture.md §3.2.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    APP_NAME: str = "Lenta Price-Tag Gateway"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    # When true (default for the skeleton) routes return mock data and never
    # touch Postgres / Redis / the ML service. Flip to false as each layer
    # gets implemented for real.
    MOCK_MODE: bool = True
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173"]

    # --- Postgres (job/result persistence) — see docs/architecture.md §3.3 ---
    DB_HOST: str = "db"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_NAME: str = "postgres"

    # --- Redis (status/list cache) ---
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_ENABLED: bool = True
    CACHE_DEFAULT_TTL: int = 60

    # --- Auth (documented §3.8, not implemented in the skeleton) ---
    JWT_SECRET_KEY: str = "change-me-in-.env"
    CSRF_SECRET_KEY: str = "change-me-in-.env"

    # --- ML inference service (the only thing the gateway calls out to) ---
    ML_BASE_URL: str = "http://ml:8002"
    ML_TIMEOUT_SECONDS: float = 600.0

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:  # Alembic uses the sync URL
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


settings = Settings()
