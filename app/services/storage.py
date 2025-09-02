import csv
import shutil
import inspect
from datetime import datetime
from typing import Dict, List, Tuple, Union, Protocol, runtime_checkable
from fastapi import HTTPException, UploadFile

from app.schemas.datasets import REQUIRED_FILES, ASSETS_COLUMNS, INCIDENTS_COLUMNS
from app.core.paths import VERSIONS_DIR, ACTIVE_DIR


@runtime_checkable
class FileLike(Protocol):
    """Interfaz mínima para soportar objetos tipo archivo en pruebas."""

    def read(self) -> bytes: ...


# === Cabeceras mínimas esperadas por archivo ===
EXPECTED_MIN_HEADERS: Dict[str, List[str]] = {
    "assets.csv": ASSETS_COLUMNS,
    "incidents.csv": INCIDENTS_COLUMNS,
}


def _read_csv_header(file_bytes: bytes) -> List[str]:
    """
    Lee y devuelve la cabecera (primera fila) de un CSV.
    - Soporta UTF-8 / UTF-8-SIG
    - Autodetecta delimitador entre coma y punto y coma
    """
    try:
        text = file_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        text = file_bytes.decode("utf-8", errors="ignore")

    sample_lines = text.splitlines()
    sample_text = "\n".join(sample_lines[:5]) if sample_lines else ""

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    reader = csv.reader(sample_lines, delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        raise HTTPException(status_code=400, detail="CSV vacío o sin cabecera")

    return [h.strip() for h in header]


def validate_csv(name: str, file_bytes: bytes) -> None:
    """
    Verifica:
      - que el archivo sea uno de los esperados (assets.csv o incidents.csv),
      - que existan todas las columnas mínimas (se permiten columnas extra).
    """
    if name not in EXPECTED_MIN_HEADERS:
        raise HTTPException(
            status_code=400,
            detail=f"Archivo inesperado: {name}. Se esperan únicamente {REQUIRED_FILES}",
        )

    got = _read_csv_header(file_bytes)
    need = EXPECTED_MIN_HEADERS[name]
    missing = [c for c in need if c not in got]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"{name}: columnas faltantes {missing}. Cabecera recibida: {got}",
        )


def create_version_id() -> str:
    return "v" + datetime.now().strftime("%Y%m%d-%H%M%S")


async def _read_any(up: Union[UploadFile, FileLike]) -> bytes:
    """
    Lee bytes desde:
      - UploadFile (async),
      - file-like con .read() -> bytes (sync),
      - file-like con .read() async (coroutine function).
    """
    # Caso UploadFile (Starlette/FastAPI)
    if isinstance(up, UploadFile):
        return await up.read()

    # Caso file-like genérico
    read = getattr(up, "read", None)
    if read is None:
        raise HTTPException(status_code=400, detail="Objeto no tiene método .read()")

    # .read es coroutine function (async)?
    if inspect.iscoroutinefunction(read):
        return await read()

    # .read es llamable sync
    if callable(read):
        return read()

    raise HTTPException(status_code=400, detail="Método .read inválido")


async def save_version(
    files: Dict[str, Union[UploadFile, FileLike]],
) -> Tuple[str, List[str]]:
    """
    Guarda una nueva versión con los archivos recibidos.
    - Requiere: assets.csv, incidents.csv
    - Valida cabeceras mínimas (se permiten columnas extra)
    - Devuelve: (version_id, lista_de_archivos_guardados)
    """
    # Verifica que estén los dos requeridos
    for required in REQUIRED_FILES:
        if required not in files:
            raise HTTPException(status_code=400, detail=f"Falta subir: {required}")

    version_id = create_version_id()
    version_dir = VERSIONS_DIR / version_id
    version_dir.mkdir(parents=True, exist_ok=False)

    saved: List[str] = []
    for name, up in files.items():
        # ✅ Lectura robusta (corrige el error de coroutine sin await)
        content = await _read_any(up)

        # Valida cabecera mínima
        validate_csv(name, content)

        # Persiste
        (version_dir / name).write_bytes(content)
        saved.append(name)

    return version_id, saved


def list_versions() -> List[Tuple[str, List[str]]]:
    """
    Lista versiones disponibles con los archivos presentes en cada una.
    """
    rows: List[Tuple[str, List[str]]] = []
    if not VERSIONS_DIR.exists():
        return rows
    for child in sorted(VERSIONS_DIR.iterdir()):
        if child.is_dir():
            files = [p.name for p in child.glob("*.csv")]
            rows.append((child.name, files))
    return rows


def activate_version(version_id: str) -> None:
    """
    Marca una versión como activa:
      - Limpia data/active
      - Copia los CSV de la versión seleccionada
    """
    src = VERSIONS_DIR / version_id
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Versión {version_id} no existe")

    # Limpia el directorio activo
    for p in ACTIVE_DIR.glob("*"):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

    # Copia todos los CSV de la versión
    for f in src.glob("*.csv"):
        shutil.copy2(f, ACTIVE_DIR / f.name)
