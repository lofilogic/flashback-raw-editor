"""
Log file in the app-data dir, so users can send us what went wrong.

Each launch starts a new file; the previous sessions are kept as
lofilogic.log.1 … .3. Nothing is uploaded: the user copies or opens it.
"""
import logging
import logging.handlers
import platform
import sys
from pathlib import Path

from PySide6.QtCore import QStandardPaths

LOG_NAME = "lofilogic.log"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUPS = 3
# Enough for a session with a few tracebacks; chat apps turn longer pastes
# into a file anyway.
_COPY_MAX_CHARS = 200_000

_log_path = None


def log_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not base:
        base = str(Path.home() / '.lofilogic')
    p = Path(base) / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_path():
    """Current session's log file, or None if file logging failed."""
    return _log_path


def setup(version: str):
    """Log to stderr (when there is one) and to the log file.

    Call after the app name is set, since that decides the app-data dir."""
    global _log_path
    fmt = logging.Formatter("%(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Windowed builds have no stderr.
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    try:
        path = log_dir() / LOG_NAME
        fh = logging.handlers.RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS,
            encoding="utf-8", delay=True)
        if path.exists() and path.stat().st_size > 0:
            fh.doRollover()
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(fh)
        _log_path = path
    except Exception:
        logging.getLogger(__name__).exception("Could not open the log file")

    sys.excepthook = _log_uncaught
    logging.getLogger(__name__).info(
        "LoFi Logic %s | %s %s | Python %s",
        version, platform.system(), platform.release(), platform.python_version())


def _log_uncaught(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logging.getLogger("uncaught").critical("Uncaught exception", exc_info=(exc_type, exc, tb))


def text_for_clipboard(header: str, previous: bool = False) -> str:
    """Header plus this session's log, or the previous session's (e.g. after
    a crash). Trimmed from the start if long."""
    if _log_path is None:
        body = "(no log file)"
    else:
        path = _log_path.with_name(LOG_NAME + ".1") if previous else _log_path
        for h in logging.getLogger().handlers:
            h.flush()
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            body = "(no previous session log)" if previous else "(log is empty)"
        except OSError as e:
            body = f"(could not read {path}: {e})"
    if len(body) > _COPY_MAX_CHARS:
        body = "… (trimmed)\n" + body[-_COPY_MAX_CHARS:]
    return f"{header}\n\n{body}"
