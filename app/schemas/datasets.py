from pydantic import BaseModel, Field
from typing import Literal, List

AssetFile = Literal["assets.csv"]
IncidentFile = Literal["incidents.csv"]

REQUIRED_FILES = ["assets.csv", "incidents.csv"]

# Mínimas (el builder acepta delivery_date en ISO o DD/MM/YYYY y 'date' como alias de incident_date)
ASSETS_COLUMNS = ["asset_id", "model", "delivery_date", "school"]
INCIDENTS_COLUMNS = [
    "asset_id",
    "failure_type",
]  # la fecha puede venir como 'incident_date' o 'date' (builder normaliza)


class IngestResult(BaseModel):
    version_id: str = Field(..., description="Identificador de versión creada")
    saved_files: List[str]


class VersionInfo(BaseModel):
    version_id: str
    files: List[str]
    is_active: bool
