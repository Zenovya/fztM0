from pydantic import BaseModel
import os

class Settings(BaseModel):
    APP_NAME: str = "MVP-IA API"
    APP_VERSION: str = "0.1.0"
    ENV: str = os.getenv("ENV", "dev")
    ALLOW_ORIGINS: list[str] = ["*"]

settings = Settings()
