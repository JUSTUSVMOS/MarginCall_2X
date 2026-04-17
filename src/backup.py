import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import DB_FILE, PROJECT_ROOT
from src.database import db_lock


logger = logging.getLogger(__name__)

BACKUP_DIR = PROJECT_ROOT / "backups"
MAX_BACKUPS = 30


def _build_backup_path(db_path: Path, backup_dir: Path, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    return backup_dir / f"{db_path.stem}_{timestamp}.db"


def _cleanup_old_backups(backup_dir: Path = BACKUP_DIR, *, stem: str = "portfolio", max_backups: int = MAX_BACKUPS):
    backups = sorted(backup_dir.glob(f"{stem}_*.db"))
    for old in backups[:-max_backups]:
        old.unlink()


def backup_database(
    db_path: Path | None = None,
    *,
    backup_dir: Path | None = None,
    max_backups: int = MAX_BACKUPS,
) -> Path | None:
    db_path = Path(db_path or DB_FILE)
    backup_dir = Path(backup_dir or BACKUP_DIR)

    if not db_path.exists():
        logger.warning(f"Database backup skipped; source DB does not exist: {db_path}")
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = _build_backup_path(db_path, backup_dir)

    try:
        with sqlite3.connect(str(db_path)) as source_conn, sqlite3.connect(str(backup_path)) as backup_conn:
            source_conn.backup(backup_conn)
    except sqlite3.Error as exc:
        logger.warning(f"SQLite online backup failed for {db_path}, falling back to file copy: {exc}")
        try:
            with db_lock:
                shutil.copy2(db_path, backup_path)
        except OSError:
            logger.exception(f"Database backup failed for {db_path}")
            return None

    _cleanup_old_backups(backup_dir=backup_dir, stem=db_path.stem, max_backups=max_backups)
    logger.info(f"✅ DB backup: {backup_path}")
    return backup_path
