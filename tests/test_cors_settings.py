import sys
from pathlib import Path

BACKEND_PATH = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.config import Settings

def test_default_cors_origin(monkeypatch):
    monkeypatch.delenv("UI_ORIGIN", raising=False)
    settings = Settings(_env_file=None)
    assert settings.cors_origins == ["http://localhost:3000"]

def test_multiple_cors_origins(monkeypatch):
    monkeypatch.setenv("UI_ORIGIN", "http://foo.local, https://bar.local")
    settings = Settings(_env_file=None)
    assert settings.cors_origins == ["http://foo.local", "https://bar.local"]
