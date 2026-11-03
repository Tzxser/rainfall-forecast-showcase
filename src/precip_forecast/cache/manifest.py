"""Deterministic cache manifests and safe filesystem primitives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_BROAD_ROOTS = {Path("/"), Path("/home"), Path("/tmp"), Path("/var")}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: str | Path) -> None:
    """Persist directory entries on POSIX; remain importable on other hosts."""

    directory = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_cache_root(
    cache_root: str | Path,
    data_root: str | Path,
    *,
    required_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve roots and fail closed on broad, nested, or implicit targets."""

    cache_argument = Path(cache_root)
    data_argument = Path(data_root)
    if not cache_argument.is_absolute() or not data_argument.is_absolute():
        raise ValueError("data_root and cache_root must be explicit absolute paths")
    cache = cache_argument.resolve()
    data = data_argument.resolve()
    if cache in FORBIDDEN_BROAD_ROOTS or len(cache.parts) < 4:
        raise ValueError(f"refusing broad cache root: {cache}")
    if cache == data or _contains(data, cache) or _contains(cache, data):
        raise ValueError("cache root and read-only data root must be disjoint")
    if required_root is not None and cache != Path(required_root).resolve():
        raise ValueError(f"cache root must be exactly {Path(required_root).resolve()}")
    return cache, data


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe cache payload path: {value!r}")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"ambiguous cache payload path: {value!r}")
    return path


def resolve_payload(cache_root: str | Path, relative: str) -> Path:
    root = Path(cache_root).resolve()
    candidate = (root / safe_relative_path(relative)).resolve()
    if not _contains(root, candidate) or candidate == root:
        raise ValueError(f"cache payload escapes root: {relative!r}")
    return candidate


def file_record(root: str | Path, path: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    source = Path(path).resolve()
    if not source.is_file() or not _contains(base, source):
        raise ValueError(f"manifest payload is not a file below cache root: {source}")
    relative = source.relative_to(base).as_posix()
    return {
        "path": relative,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def render_sha256s(records: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    seen: set[str] = set()
    for record in sorted(records, key=lambda item: str(item["path"])):
        path = safe_relative_path(str(record["path"])).as_posix()
        digest = str(record["sha256"]).lower()
        if path in seen or not SHA256_RE.fullmatch(digest):
            raise ValueError("invalid or duplicate SHA256 manifest record")
        seen.add(path)
        rows.append(f"{digest}  {path}")
    if not rows:
        raise ValueError("SHA256SUMS cannot be empty")
    return "\n".join(rows) + "\n"


def content_identifier(
    *,
    preprocess_config_sha256: str,
    source_fingerprint_sha256: str,
    data_fingerprint_sha256: str,
    normalization_sha256: str,
    payload_records: Iterable[Mapping[str, Any]],
) -> str:
    """Return the repeatable cache identity, excluding wall-clock metadata."""

    deterministic = {
        "schema_version": 1,
        "preprocess_config_sha256": preprocess_config_sha256,
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "data_fingerprint_sha256": data_fingerprint_sha256,
        "normalization_sha256": normalization_sha256,
        "payloads": [
            {
                "path": str(record["path"]),
                "size_bytes": int(record["size_bytes"]),
                "sha256": str(record["sha256"]),
            }
            for record in sorted(payload_records, key=lambda item: str(item["path"]))
        ],
    }
    return sha256_bytes(canonical_json_bytes(deterministic))
