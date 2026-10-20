# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Verified immutable response entries for bounded official NRW requests.

The cache performs no retries. Successful entries are published atomically and
never replaced. Locks coordinate threads in this process; separate processes
may make duplicate requests, but cannot overwrite a published entry.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
from urllib.parse import urlsplit
import uuid

from . import nrw
from .pv4ger import DatasetImportError
from .validation import _invalid_constant, _unique_object

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_HEADER_BYTES = 32 * 1024
FIELDS = {"cache_schema_version", "url", "url_sha256", "response_sha256", "response_bytes",
          "headers", "fetched_at_utc", "maximum_bytes_at_fetch"}
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[tuple[str, str], threading.Lock] = {}


class AcquisitionCacheError(DatasetImportError):
    """A request or cache entry cannot be used with verified bounded evidence."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _url(url: str) -> None:
    if not isinstance(url, str) or not 1 <= len(url.encode("utf-8")) <= 16 * 1024:
        raise AcquisitionCacheError("Expected a bounded official request URL.")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.fragment or parsed.username or parsed.password
            or parsed.port is not None or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url)
            or (parsed.hostname, parsed.path) not in {
                ("ogc-api.nrw.de", "/lika/v1/collections/gebaeude_bauwerk/items"),
                ("www.wms.nrw.de", "/geobasis/wms_nw_dop"),
            }):
        raise AcquisitionCacheError("Only the fixed official NRW building-items and DOP endpoints are cached.")


def _headers(headers: dict) -> dict:
    if (not isinstance(headers, dict) or len(headers) > 128
            or any(not isinstance(key, str) or not isinstance(value, str)
                   or not 1 <= len(key) <= 256 or len(value) > 8192 for key, value in headers.items())
            or len(json.dumps(headers, ensure_ascii=False).encode("utf-8")) > MAX_HEADER_BYTES):
        raise AcquisitionCacheError("Official response headers exceed their bounded string mapping contract.")
    return dict(headers)


def _root(cache_dir: Path | str) -> Path:
    path = Path(cache_dir).absolute()
    if path.is_symlink():
        raise AcquisitionCacheError("Cache root must not be a symbolic link.")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise AcquisitionCacheError("Expected a regular cache directory.")
    # Resolve the caller-selected root once (including OS aliases such as /var).
    # Untrusted entry names are never used: each child name is a URL SHA-256.
    return path.resolve(strict=True)


def _read_regular(directory_fd: int, name: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise AcquisitionCacheError("Cache payload must be a bounded regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum + 1)
        if len(payload) > maximum or len(payload) != info.st_size:
            raise AcquisitionCacheError("Cache payload changed or exceeded its byte bound.")
        return payload
    finally:
        os.close(descriptor)


def _load(root: Path, key: str, url: str, maximum: int) -> tuple[bytes, dict, dict]:
    descriptor = os.open(root / key, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if set(os.listdir(descriptor)) != {"response.bin", "metadata.json"}:
            raise AcquisitionCacheError("Cache entry must contain exactly its response and metadata.")
        metadata_bytes = _read_regular(descriptor, "metadata.json", MAX_METADATA_BYTES)
        metadata = json.loads(metadata_bytes, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        if (not isinstance(metadata, dict) or set(metadata) != FIELDS
                or metadata["cache_schema_version"] != "1.0" or metadata["url"] != url
                or metadata["url_sha256"] != key or type(metadata["response_bytes"]) is not int
                or type(metadata["maximum_bytes_at_fetch"]) is not int
                or not 0 <= metadata["response_bytes"] <= maximum
                or not max(1, metadata["response_bytes"]) <= metadata["maximum_bytes_at_fetch"] <= MAX_RESPONSE_BYTES
                or not isinstance(metadata["response_sha256"], str)
                or len(metadata["response_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in metadata["response_sha256"])):
            raise AcquisitionCacheError("Cache metadata does not bind the requested URL, size or schema.")
        timestamp = datetime.fromisoformat(metadata["fetched_at_utc"])
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise AcquisitionCacheError("Cache retrieval timestamp must be timezone-aware UTC.")
        headers = _headers(metadata["headers"])
        payload = _read_regular(descriptor, "response.bin", maximum)
        if len(payload) != metadata["response_bytes"] or _digest(payload) != metadata["response_sha256"]:
            raise AcquisitionCacheError("Cache response hash or declared size differs; no network request was made.")
        return payload, headers, {**metadata, "metadata_sha256": _digest(metadata_bytes),
                                  "cache_hit": True, "network_request_performed": False, "cache_entry": key}
    finally:
        os.close(descriptor)


def _write_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _failure(root: Path, key: str, url: str, error: BaseException) -> None:
    directory = root / "failures"
    if directory.is_symlink():
        raise AcquisitionCacheError("Cache failure-log directory must not be a symbolic link.")
    directory.mkdir(exist_ok=True)
    record = {"url": url, "url_sha256": key, "failed_at_utc": datetime.now(timezone.utc).isoformat(),
              "error_type": type(error).__name__, "error": str(error)[:2048]}
    token = uuid.uuid4().hex
    temporary = directory / f".{key}-{token}.tmp"
    final = directory / f"{key}-{token}.json"
    try:
        _write_file(temporary, (json.dumps(record, sort_keys=True) + "\n").encode())
        os.rename(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)


def cached_fetch(url: str, maximum: int, *, cache_dir: Path | str,
                 fetcher=None) -> tuple[bytes, dict[str, str], dict]:
    """Return original bytes, response headers and per-response cache evidence.

    A cache hit retains the original ``fetched_at_utc``. ``cache_hit`` identifies
    entry reuse; ``network_request_performed`` separately records this call's
    fetch, including a publication race between processes. Corrupt entries fail without silent redownload.
    Failed/interrupted fetches are logged but never published as cache entries.
    Each call performs at most one fetch; retry policy belongs to its caller.
    """
    stage = None
    try:
        _url(url)
        if type(maximum) is not int or not 1 <= maximum <= MAX_RESPONSE_BYTES:
            raise AcquisitionCacheError(f"Maximum response bytes must be an integer from 1 to {MAX_RESPONSE_BYTES}.")
        if fetcher is not None and not callable(fetcher):
            raise AcquisitionCacheError("Fetcher must be callable.")
        root = _root(cache_dir)
        key = _digest(url.encode("utf-8"))
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault((str(root), key), threading.Lock())
        with lock:
            entry = root / key
            if os.path.lexists(entry):
                return _load(root, key, url, maximum)
            try:
                payload, headers = (fetcher or nrw._fetch)(url, maximum)
                if not isinstance(payload, bytes) or len(payload) > maximum:
                    raise AcquisitionCacheError("Fetcher returned non-byte or oversized response data.")
                headers = _headers(headers)
                metadata = {"cache_schema_version": "1.0", "url": url, "url_sha256": key,
                            "response_sha256": _digest(payload), "response_bytes": len(payload), "headers": headers,
                            "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "maximum_bytes_at_fetch": maximum}
                metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
                if len(metadata_bytes) > MAX_METADATA_BYTES:
                    raise AcquisitionCacheError("Response metadata exceeds its byte bound.")
                stage = Path(tempfile.mkdtemp(prefix=f".{key[:16]}-response-", dir=root))
                _write_file(stage / "response.bin", payload)
                _write_file(stage / "metadata.json", metadata_bytes)
                try:
                    os.rename(stage, entry)
                except OSError:
                    if not os.path.lexists(entry):
                        raise
                    # Another process won the publication race. It cannot be
                    # overwritten; validate and use that complete entry.
                    cached_payload, cached_headers, evidence = _load(root, key, url, maximum)
                    return cached_payload, cached_headers, {**evidence, "network_request_performed": True,
                                                           "publication_race": True,
                                                           "discarded_response_sha256": _digest(payload)}
                stage = None
                return payload, headers, {**metadata, "metadata_sha256": _digest(metadata_bytes),
                                          "cache_hit": False, "network_request_performed": True, "cache_entry": key}
            except BaseException as error:
                try:
                    _failure(root, key, url, error)
                except (OSError, ValueError) as logging_error:
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise error
                    raise AcquisitionCacheError(f"Acquisition failed and failure evidence could not be recorded: {logging_error}") from error
                raise
    except AcquisitionCacheError:
        raise
    except Exception as error:
        raise AcquisitionCacheError(f"Official response cache failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
