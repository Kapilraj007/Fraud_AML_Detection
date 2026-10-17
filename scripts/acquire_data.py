#!/usr/bin/env python3
"""Governed, dependency-free acquisition for the Sprint 1 data registry.

Every successful fetch is streamed to a partial file, verified, atomically
promoted, and recorded in a provenance manifest. The tool never accepts
Kaggle competition rules on a user's behalf.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "datasets.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "finance-fraud-aml-data-acquisition/1.0"
KAGGLE_HOSTS = frozenset({"kaggle.com", "www.kaggle.com"})
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


class AcquisitionError(RuntimeError):
    """Base class for a controlled acquisition failure."""


class RegistryError(AcquisitionError):
    """The registry is malformed or a requested entry does not exist."""


class IntegrityError(AcquisitionError):
    """A file differs from its governed digest."""


class UnsafeArchiveError(AcquisitionError):
    """An archive could write outside its destination or create a link."""


class CredentialError(AcquisitionError):
    """Required Kaggle credentials are absent or invalid."""


@dataclass(frozen=True)
class KaggleCredentials:
    scheme: str
    secret: str
    source: str
    username: str | None = None


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is None:
        port = 443 if parsed.scheme.casefold() == "https" else 80
    return parsed.scheme.casefold(), parsed.hostname, port


def _same_https_origin(first_url: str, second_url: str) -> bool:
    first = _origin(first_url)
    second = _origin(second_url)
    return first[0] == "https" and first == second


def _is_kaggle_https_url(url: str) -> bool:
    scheme, hostname, port = _origin(url)
    return scheme == "https" and hostname in KAGGLE_HOSTS and port == 443


class _NoCrossHostAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward a Kaggle Authorization header to signed storage URLs."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        if not _same_https_origin(req.full_url, newurl):
            redirected.remove_header("Authorization")
        return redirected


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = Path(path)
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"Dataset registry not found: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Invalid JSON in {registry_path}: {exc}") from exc

    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("datasets"), dict):
        raise RegistryError("Registry must have schema_version 1 and a datasets object")
    for dataset_id, dataset in data["datasets"].items():
        _validate_dataset_id(dataset_id)
        if not isinstance(dataset, dict) or not isinstance(dataset.get("files"), list):
            raise RegistryError(f"Dataset {dataset_id!r} must contain a files list")
        _validate_dataset_file_names(dataset_id, dataset["files"])
        _validate_dataset_access(dataset_id, dataset)
    return data


def _validate_dataset_id(dataset_id: str) -> None:
    if not isinstance(dataset_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset_id):
        raise RegistryError(f"Unsafe dataset ID: {dataset_id!r}")


def _validate_dataset_file_names(dataset_id: str, files: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise RegistryError(f"Every file in dataset {dataset_id!r} must be an object")
        name = item.get("name")
        windows_path = PureWindowsPath(name) if isinstance(name, str) else None
        reserved_stem = windows_path.stem.upper() if windows_path else ""
        windows_reserved = reserved_stem in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))
        }
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or windows_path.name != name
            or windows_path.drive
            or ":" in name
            or name.casefold() == "manifest.json"
            or name.casefold().endswith((".part", ".extracted.part"))
            or windows_reserved
        ):
            raise RegistryError(f"Unsafe file name in dataset {dataset_id!r}: {name!r}")
        folded = name.casefold()
        if folded in seen:
            raise RegistryError(f"Duplicate file name in dataset {dataset_id!r}: {name}")
        seen.add(folded)
        url = item.get("url")
        if not isinstance(url, str) or urllib.parse.urlsplit(url).scheme not in {"https", "file"}:
            raise RegistryError(f"File {name!r} must use an https:// or file:// URL")
        expected = item.get("sha256")
        if expected is not None and (
            not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None
        ):
            raise RegistryError(f"File {name!r} has an invalid SHA-256 value")


def _validate_dataset_access(dataset_id: str, dataset: Mapping[str, Any]) -> None:
    access_type = dataset.get("access_type")
    if not isinstance(access_type, str):
        raise RegistryError(f"Dataset {dataset_id!r} must define access_type")
    if access_type.startswith("kaggle_"):
        for item in dataset["files"]:
            if not _is_kaggle_https_url(str(item["url"])):
                raise RegistryError(
                    f"Authenticated Kaggle dataset {dataset_id!r} has a non-Kaggle URL: {item['url']}"
                )


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path | str, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual.casefold() != expected_sha256.casefold():
        raise IntegrityError(
            f"SHA-256 mismatch for {Path(path)}: expected {expected_sha256}, got {actual}"
        )
    return actual


def _credential_paths() -> list[Path]:
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    paths: list[Path] = []
    if configured:
        candidate = Path(configured).expanduser()
        paths.append(candidate if candidate.name == "kaggle.json" else candidate / "kaggle.json")
    default = Path.home() / ".kaggle" / "kaggle.json"
    if default not in paths:
        paths.append(default)
    return paths


def _access_token_paths() -> list[Path]:
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    paths: list[Path] = []
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.name in {"access_token", "access_token.txt"}:
            paths.append(candidate)
        else:
            paths.extend((candidate / "access_token", candidate / "access_token.txt"))
    for name in ("access_token", "access_token.txt"):
        default = Path.home() / ".kaggle" / name
        if default not in paths:
            paths.append(default)
    return paths


def find_kaggle_credentials() -> KaggleCredentials | None:
    api_token = os.environ.get("KAGGLE_API_TOKEN")
    if api_token:
        return KaggleCredentials("bearer", api_token, "KAGGLE_API_TOKEN")

    for path in _access_token_paths():
        if path.is_file():
            try:
                token = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise CredentialError(f"Cannot read Kaggle access token from {path}: {exc}") from exc
            if not token:
                raise CredentialError(f"Kaggle access token is empty: {path}")
            return KaggleCredentials("bearer", token, str(path))

    env_username = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_username or env_key:
        if not (env_username and env_key):
            raise CredentialError(
                "Both KAGGLE_USERNAME and KAGGLE_KEY must be set; only one was found"
            )
        return KaggleCredentials("basic", env_key, "environment", env_username)

    for path in _credential_paths():
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CredentialError(f"Cannot read Kaggle credentials from {path}: {exc}") from exc
        username = payload.get("username")
        key = payload.get("key")
        if not isinstance(username, str) or not isinstance(key, str) or not username or not key:
            raise CredentialError(
                f"{path} must contain non-empty JSON fields 'username' and 'key'"
            )
        return KaggleCredentials("basic", key, str(path), username)
    return None


def credential_status() -> dict[str, Any]:
    try:
        credentials = find_kaggle_credentials()
    except CredentialError as exc:
        return {"available": False, "source": None, "error": str(exc)}
    if credentials is None:
        return {"available": False, "source": None, "error": None}
    return {"available": True, "source": credentials.source, "error": None}


def _authorization_headers(credentials: KaggleCredentials | None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/octet-stream"}
    if credentials is not None:
        if credentials.scheme == "bearer":
            headers["Authorization"] = f"Bearer {credentials.secret}"
        else:
            token = base64.b64encode(
                f"{credentials.username}:{credentials.secret}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
    return headers


def _normalised_zip_name(name: str) -> PurePosixPath:
    if "\x00" in name:
        raise UnsafeArchiveError("Archive member contains a NUL byte")
    portable_name = name.replace("\\", "/")
    posix_path = PurePosixPath(portable_name)
    windows_path = PureWindowsPath(name)
    if (
        not portable_name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
    ):
        raise UnsafeArchiveError(f"Unsafe archive member path: {name!r}")
    return posix_path


def _validated_zip_members(
    archive: zipfile.ZipFile, destination: Path
) -> list[tuple[zipfile.ZipInfo, Path]]:
    root = destination.resolve()
    validated: list[tuple[zipfile.ZipInfo, Path]] = []
    seen: set[str] = set()
    for info in archive.infolist():
        portable = _normalised_zip_name(info.filename)
        target = (root / Path(*portable.parts)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise UnsafeArchiveError(f"Archive member escapes destination: {info.filename!r}") from exc
        unix_mode = info.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise UnsafeArchiveError(f"Archive member is a symbolic link: {info.filename!r}")
        key = str(target).casefold()
        if key in seen:
            raise UnsafeArchiveError(f"Archive contains colliding paths: {info.filename!r}")
        seen.add(key)
        validated.append((info, target))
    return validated


def _open_private_binary(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "wb")


def safe_extract_zip(archive_path: Path | str, destination: Path | str) -> list[Path]:
    """Extract a ZIP only after every member passes traversal/link checks."""

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            validated = _validated_zip_members(archive, destination_path)
            for info, target in validated:
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, _open_private_binary(target) as sink:
                    shutil.copyfileobj(source, sink, length=CHUNK_SIZE)
                extracted.append(target)
    except zipfile.BadZipFile as exc:
        raise UnsafeArchiveError(f"Invalid ZIP archive: {archive_path}") from exc
    return extracted


def _extract_expected_member(
    archive_path: Path, expected_member: str, output_part: Path
) -> None:
    """Validate the complete ZIP, then stream exactly one expected member."""

    try:
        with zipfile.ZipFile(archive_path) as archive:
            validated = _validated_zip_members(archive, output_part.parent)
            exact = [pair for pair in validated if pair[0].filename == expected_member]
            candidates = exact or [
                pair
                for pair in validated
                if _normalised_zip_name(pair[0].filename).name == expected_member
            ]
            candidates = [pair for pair in candidates if not pair[0].is_dir()]
            if len(candidates) != 1:
                raise UnsafeArchiveError(
                    f"Expected exactly one {expected_member!r} member; found {len(candidates)}"
                )
            with archive.open(candidates[0][0]) as source, _open_private_binary(output_part) as sink:
                shutil.copyfileobj(source, sink, length=CHUNK_SIZE)
    except zipfile.BadZipFile as exc:
        raise UnsafeArchiveError(f"Invalid ZIP archive from {archive_path}") from exc


def _open_url(url: str, headers: Mapping[str, str], timeout: float) -> BinaryIO:
    if "Authorization" in headers and not _is_kaggle_https_url(url):
        raise CredentialError("Refusing to send Kaggle credentials to a non-Kaggle HTTPS origin")
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    opener = urllib.request.build_opener(_NoCrossHostAuthRedirectHandler())
    try:
        return opener.open(request, timeout=timeout)  # type: ignore[return-value]
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise CredentialError(
                f"Kaggle denied access to {url} (HTTP {exc.code}). Configure credentials; "
                "for competition data, first join the competition and accept its rules in a browser."
            ) from exc
        raise AcquisitionError(f"Download failed for {url}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise AcquisitionError(f"Download failed for {url}: {exc.reason}") from exc


def _stream_url(url: str, part_path: Path, headers: Mapping[str, str], timeout: float) -> int:
    total = 0
    with _open_url(url, headers, timeout) as response, _open_private_binary(part_path) as output:
        while True:
            block = response.read(CHUNK_SIZE)
            if not block:
                break
            output.write(block)
            total += len(block)
        output.flush()
        os.fsync(output.fileno())
    if total == 0:
        raise AcquisitionError(f"Source returned an empty file: {url}")
    return total


def _looks_like_html(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(1024).lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _cleanup_parts(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _acquire_file(
    file_entry: Mapping[str, Any],
    destination: Path,
    headers: Mapping[str, str],
    *,
    force: bool,
    timeout: float,
    trusted_existing_sha256: str | None = None,
) -> dict[str, Any] | None:
    name = str(file_entry["name"])
    target = destination / name
    expected = file_entry.get("sha256")
    baseline = expected or trusted_existing_sha256
    if target.is_symlink():
        raise AcquisitionError(f"Refusing to read or replace symbolic-link target: {target}")
    if target.exists() and not force:
        if not trusted_existing_sha256:
            raise AcquisitionError(
                f"Existing file {target} has no provenance manifest record; "
                "use --force to reacquire and manifest it"
            )
        verify_file(target, str(baseline))
        print(f"SKIP {target} (already exists; use --force to replace)")
        return None

    download_part = target.with_name(target.name + ".part")
    extracted_part = target.with_name(target.name + ".extracted.part")
    _cleanup_parts(download_part, extracted_part)
    fetched_at = utc_now()
    try:
        _stream_url(str(file_entry["url"]), download_part, headers, timeout)
        candidate = download_part
        if zipfile.is_zipfile(download_part):
            member = str(file_entry.get("archive_member") or name)
            _extract_expected_member(download_part, member, extracted_part)
            candidate = extracted_part
        elif _looks_like_html(download_part):
            raise AcquisitionError(
                f"{file_entry['url']} returned HTML instead of {name}; Kaggle login may be required"
            )

        actual = sha256_file(candidate)
        if baseline and actual.casefold() != str(baseline).casefold():
            raise IntegrityError(
                f"SHA-256 mismatch for {name}: expected {baseline}, got {actual}; file not installed. "
                "Review upstream changes before rotating a governed digest."
            )
        size = candidate.stat().st_size
        os.replace(candidate, target)
        if download_part != candidate:
            _cleanup_parts(download_part)
        print(f"OK   {target} ({size} bytes, sha256={actual})")
        return {
            "name": name,
            "role": file_entry.get("role"),
            "source_url": file_entry["url"],
            "fetched_at": fetched_at,
            "bytes": size,
            "sha256": actual,
            "expected_sha256": expected,
        }
    except Exception:
        _cleanup_parts(download_part, extracted_part)
        raise


def _read_manifest(
    path: Path,
    dataset_id: str | None = None,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"Cannot read existing manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise AcquisitionError(f"Existing manifest is malformed: {path}")
    if dataset_id is None or dataset is None:
        return payload
    if (
        payload.get("schema_version") != 1
        or payload.get("dataset_id") != dataset_id
        or payload.get("dataset_version") != dataset.get("version")
    ):
        raise AcquisitionError(
            f"Manifest identity/version does not match registry dataset {dataset_id!r}: {path}"
        )
    registry_files = {str(item["name"]): item for item in dataset["files"]}
    seen: set[str] = set()
    for item in payload["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise AcquisitionError(f"Manifest contains a malformed file record: {path}")
        name = item["name"]
        if name in seen or name not in registry_files:
            raise AcquisitionError(f"Manifest contains an unknown/duplicate file {name!r}: {path}")
        seen.add(name)
        sha256 = item.get("sha256")
        expected_source = registry_files[name]["url"]
        if (
            not isinstance(sha256, str)
            or SHA256_PATTERN.fullmatch(sha256) is None
            or item.get("source_url") != expected_source
            or item.get("dataset_version") != dataset.get("version")
            or item.get("license") != dataset["license"]["id"]
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or not isinstance(item.get("fetched_at"), str)
            or not item["fetched_at"].endswith("Z")
        ):
            raise AcquisitionError(f"Manifest record for {name!r} is invalid or stale: {path}")
    return payload


def _write_manifest(
    path: Path,
    dataset_id: str,
    dataset: Mapping[str, Any],
    new_record: Mapping[str, Any],
    license_acknowledged: bool,
) -> None:
    existing = _read_manifest(path, dataset_id, dataset)
    records: dict[str, dict[str, Any]] = {}
    if existing:
        records = {
            str(item["name"]): dict(item)
            for item in existing["files"]
            if isinstance(item, dict) and "name" in item
        }
    license_info = dict(dataset["license"])
    enriched = dict(new_record)
    enriched.update(
        {
            "dataset_version": dataset["version"],
            "license": license_info["id"],
            "license_url": license_info["url"],
            "license_acknowledged": license_acknowledged,
        }
    )
    records[str(enriched["name"])] = enriched
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "dataset_version": dataset["version"],
        "provider": dataset["provider"],
        "landing_page": dataset["landing_page"],
        "license": license_info,
        "updated_at": utc_now(),
        "files": [records[name] for name in sorted(records, key=str.casefold)],
    }
    part = path.with_name(path.name + ".part")
    _cleanup_parts(part)
    try:
        descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, path)
    except Exception:
        _cleanup_parts(part)
        raise


def _select_files(dataset: Mapping[str, Any], names: Sequence[str] | None) -> list[dict[str, Any]]:
    available = {str(item["name"]): item for item in dataset["files"]}
    if not names:
        return list(dataset["files"])
    unknown = [name for name in names if name not in available]
    if unknown:
        raise RegistryError(
            f"Unknown file(s): {', '.join(unknown)}. Available: {', '.join(available)}"
        )
    return [available[name] for name in names]


def _dataset_destination(output_dir: Path | str, dataset_id: str, *, create: bool) -> Path:
    root = Path(output_dir).resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    candidate = root / dataset_id
    is_junction = getattr(candidate, "is_junction", lambda: False)
    if candidate.is_symlink() or is_junction():
        raise AcquisitionError(f"Refusing linked dataset directory: {candidate}")
    if create:
        candidate.mkdir(mode=0o700, parents=False, exist_ok=True)
    destination = candidate.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise AcquisitionError(f"Dataset directory escapes output root: {destination}") from exc
    if create and os.name != "nt":
        os.chmod(destination, 0o700)
    return destination


def download_dataset(
    registry: Mapping[str, Any],
    dataset_id: str,
    output_dir: Path | str,
    *,
    selected_files: Sequence[str] | None = None,
    accept_license: bool = False,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    _validate_dataset_id(dataset_id)
    try:
        dataset = registry["datasets"][dataset_id]
    except KeyError as exc:
        raise RegistryError(f"Unknown dataset: {dataset_id}") from exc
    _validate_dataset_file_names(dataset_id, dataset["files"])
    _validate_dataset_access(dataset_id, dataset)

    license_info = dataset["license"]
    if license_info.get("acknowledgement_required") and not accept_license:
        raise AcquisitionError(
            f"{dataset['name']} is governed by {license_info['id']}. Read {license_info['url']} "
            "and rerun with --accept-license to record your acknowledgement."
        )

    access_type = dataset.get("access_type")
    credentials = find_kaggle_credentials() if str(access_type).startswith("kaggle_") else None
    if dataset.get("credentials_required") and credentials is None:
        rules = dataset.get("rules_page", dataset["distribution_page"])
        raise CredentialError(
            "Kaggle credentials were not found. Set KAGGLE_API_TOKEN, set KAGGLE_USERNAME and "
            "KAGGLE_KEY together, or configure ~/.kaggle/access_token or kaggle.json. "
            "IEEE-CIS additionally requires you to join the "
            f"competition and accept its rules in a browser: {rules}"
        )
    if dataset.get("access_type") == "kaggle_competition":
        print(
            "NOTICE: --accept-license only records local acknowledgement. Kaggle must already show "
            f"that your account accepted {dataset['rules_page']}."
        )

    destination = _dataset_destination(output_dir, dataset_id, create=True)
    headers = _authorization_headers(credentials)
    manifest_path = destination / "manifest.json"
    existing_manifest = _read_manifest(manifest_path, dataset_id, dataset)
    existing_hashes = {
        str(item["name"]): item.get("sha256")
        for item in (existing_manifest or {}).get("files", [])
        if isinstance(item, dict) and "name" in item
    }
    for entry in _select_files(dataset, selected_files):
        record = _acquire_file(
            entry,
            destination,
            headers,
            force=force,
            timeout=timeout,
            trusted_existing_sha256=existing_hashes.get(entry["name"]),
        )
        if record is not None:
            _write_manifest(manifest_path, dataset_id, dataset, record, accept_license)
    return manifest_path


def verify_dataset(
    registry: Mapping[str, Any],
    dataset_id: str,
    output_dir: Path | str,
    *,
    selected_files: Sequence[str] | None = None,
) -> bool:
    _validate_dataset_id(dataset_id)
    try:
        dataset = registry["datasets"][dataset_id]
    except KeyError as exc:
        raise RegistryError(f"Unknown dataset: {dataset_id}") from exc
    _validate_dataset_file_names(dataset_id, dataset["files"])
    _validate_dataset_access(dataset_id, dataset)
    destination = _dataset_destination(output_dir, dataset_id, create=False)
    manifest = _read_manifest(destination / "manifest.json", dataset_id, dataset)
    manifest_hashes = {
        item["name"]: item.get("sha256")
        for item in (manifest or {}).get("files", [])
        if isinstance(item, dict) and "name" in item
    }
    success = True
    for entry in _select_files(dataset, selected_files):
        path = destination / entry["name"]
        if not path.is_file():
            print(f"MISS {path}")
            success = False
            continue
        expected = entry.get("sha256") or manifest_hashes.get(entry["name"])
        if not expected:
            print(f"UNPINNED {path} (sha256={sha256_file(path)})")
            success = False
            continue
        try:
            actual = verify_file(path, str(expected))
        except IntegrityError as exc:
            print(f"FAIL {exc}")
            success = False
        else:
            source = "registry" if entry.get("sha256") else "manifest"
            print(f"OK   {path} (sha256={actual}, baseline={source})")
    return success


def _print_list(registry: Mapping[str, Any], as_json: bool) -> None:
    status = credential_status()
    if as_json:
        result = {
            "kaggle_credentials": status,
            "datasets": [
                {
                    "id": dataset_id,
                    "name": dataset["name"],
                    "version": dataset["version"],
                    "license": dataset["license"]["id"],
                    "access_type": dataset["access_type"],
                    "credentials_required": dataset["credentials_required"],
                    "files": [entry["name"] for entry in dataset["files"]],
                }
                for dataset_id, dataset in registry["datasets"].items()
            ],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    credential_text = (
        f"available ({status['source']})" if status["available"] else "not found"
    )
    if status.get("error"):
        credential_text += f"; {status['error']}"
    print(f"Kaggle credentials: {credential_text}")
    for dataset_id, dataset in registry["datasets"].items():
        requirement = "required" if dataset["credentials_required"] else "optional"
        print(
            f"{dataset_id}: {dataset['name']} | {dataset['license']['id']} | "
            f"{dataset['access_type']} (credentials {requirement})"
        )
        for entry in dataset["files"]:
            pin = entry.get("sha256") or "manifest-on-download"
            print(f"  - {entry['name']} [{entry['role']}] sha256={pin}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY, help="Path to datasets.json"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List governed datasets and access status")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    for command, help_text in (
        ("verify", "Verify local files against registry or manifest SHA-256 values"),
        ("download", "Download and record governed dataset files"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("dataset", help="Dataset ID from the registry")
        command_parser.add_argument(
            "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Raw-data root"
        )
        command_parser.add_argument(
            "--file",
            action="append",
            dest="files",
            help="Acquire/verify one named file; repeat for multiple files",
        )
        if command == "download":
            command_parser.add_argument(
                "--accept-license",
                action="store_true",
                help="Confirm that you read and accept the dataset license/rules",
            )
            command_parser.add_argument(
                "--force", action="store_true", help="Atomically replace existing files"
            )
            command_parser.add_argument(
                "--timeout", type=float, default=60.0, help="Per-request timeout in seconds"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        if args.command == "list":
            _print_list(registry, args.json)
            return 0
        if args.command == "verify":
            return 0 if verify_dataset(
                registry, args.dataset, args.output_dir, selected_files=args.files
            ) else 1
        download_dataset(
            registry,
            args.dataset,
            args.output_dir,
            selected_files=args.files,
            accept_license=args.accept_license,
            force=args.force,
            timeout=args.timeout,
        )
        return 0
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
