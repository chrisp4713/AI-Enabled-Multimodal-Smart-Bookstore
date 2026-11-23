"""
Shared data layer: every file read/write in the app goes through here.   NFR-09, DC-03

All data lives in local text files under data/ (DC-03). Writes never edit a
file in place: they go to a temporary file in the same folder first and then
replace the original with os.replace(), which is atomic. A crash in the middle
of a write leaves the old file intact instead of a half-written one (NFR-09).

Public functions (for the whole team)
-------------------------------------
    load_json(name, default)        read data/<name>; default if missing
    save_json(name, data)           atomic write of data/<name>
    read_csv(name)                  list of dict rows (empty if missing)
    append_csv(name, row, fields)   add one row (atomic rewrite)
    write_csv(name, rows, fields)   replace the whole file (atomic)
    data_path(name)                 full path of a file in data/
    set_data_dir(path)              point the store somewhere else (tests)

File names used by the team (data contract, Appendix A of the SRS):
    users.json, books.json, purchases.csv, ratings.csv, browsing.csv,
    taboo_words.txt, order_lists.json
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# One lock for the whole store: the GUI runs the LLM and camera on worker
# threads, so two threads could otherwise write the same file at once.
_LOCK = threading.RLock()


def set_data_dir(path: str | os.PathLike) -> None:
    """Point the store at another folder (used by the tests)."""
    global DATA_DIR
    DATA_DIR = Path(path)


def data_path(name: str) -> Path:
    return DATA_DIR / name


def _atomic_write_text(path: Path, text: str) -> None:
    """Write to a temp file in the same folder, then swap it in (NFR-09)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------
def load_json(name: str, default: Any = None) -> Any:
    """Return the parsed contents of data/<name>, or `default` if it is missing."""
    path = data_path(name)
    with _LOCK:
        if not path.exists():
            return default
        with open(path, encoding="utf-8") as f:
            return json.load(f)


def save_json(name: str, data: Any) -> None:
    """Atomically replace data/<name> with `data` as pretty JSON."""
    with _LOCK:
        _atomic_write_text(data_path(name), json.dumps(data, indent=2) + "\n")


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def read_csv(name: str) -> list[dict[str, str]]:
    """Return every row of data/<name> as a dict (all values are strings)."""
    path = data_path(name)
    with _LOCK:
        if not path.exists():
            return []
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Atomically replace data/<name> with `rows`."""
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    with _LOCK:
        _atomic_write_text(data_path(name), buf.getvalue())


def append_csv(name: str, row: dict, fieldnames: list[str]) -> None:
    """Add one row to data/<name>, creating it with a header if needed."""
    with _LOCK:
        rows = read_csv(name)
        rows.append(row)
        write_csv(name, rows, fieldnames)
