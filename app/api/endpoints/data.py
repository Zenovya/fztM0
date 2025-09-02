from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from app.core.paths import ACTIVE_DIR
from app.services.storage import save_version, list_versions, activate_version
from app.schemas.datasets import IngestResult, VersionInfo, REQUIRED_FILES

router = APIRouter(prefix="/data", tags=["data"])


@router.post(
    "/ingest",
    response_model=IngestResult,
    summary="Subir y versionar CSVs (solo assets.csv e incidents.csv)",
)
async def ingest(
    assets: UploadFile = File(..., description="assets.csv"),
    incidents: UploadFile = File(..., description="incidents.csv"),
):
    files = {"assets.csv": assets, "incidents.csv": incidents}
    version_id, saved = await save_version(files)
    return IngestResult(version_id=version_id, saved_files=saved)


@router.get("/versions", response_model=list[VersionInfo], summary="Listar versiones")
def versions(active: Optional[bool] = None):
    rows = list_versions()
    active_files = {p.name for p in ACTIVE_DIR.glob("*.csv")}
    out = []
    for vid, files in rows:
        required_ok = all(req in files for req in REQUIRED_FILES)
        is_active = required_ok and set(files) == set(active_files)
        if active is None or active == is_active:
            out.append(VersionInfo(version_id=vid, files=files, is_active=is_active))
    return out


@router.post("/activate", summary="Marcar una versión como activa")
def activate(version_id: str = Form(...)):
    activate_version(version_id)
    return {"ok": True, "version_id": version_id}
