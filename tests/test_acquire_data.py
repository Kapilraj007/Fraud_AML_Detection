import hashlib
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import acquire_data


class HashVerificationTests(unittest.TestCase):
    def test_hash_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "payload.csv"
            payload = b"id,value\n1,42\n"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            self.assertEqual(acquire_data.verify_file(path, expected), expected)

    def test_hash_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "payload.csv"
            path.write_bytes(b"tampered")

            with self.assertRaises(acquire_data.IntegrityError):
                acquire_data.verify_file(path, "0" * 64)

    def test_rejects_dataset_id_path_traversal(self) -> None:
        registry = {"schema_version": 1, "datasets": {"../escape": {"files": []}}}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "registry.json"
            path.write_text(json.dumps(registry), encoding="utf-8")

            with self.assertRaises(acquire_data.RegistryError):
                acquire_data.load_registry(path)


class ArchiveSafetyTests(unittest.TestCase):
    @staticmethod
    def _write_zip(archive_path: Path, member: str, payload: bytes = b"bad") -> None:
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(member, payload)

    def test_rejects_parent_and_windows_traversal(self) -> None:
        unsafe_names = ("../escape.txt", "folder/../../escape.txt", "..\\escape.txt")
        for index, unsafe_name in enumerate(unsafe_names):
            with self.subTest(member=unsafe_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                archive_path = root / f"unsafe-{index}.zip"
                destination = root / "extract"
                self._write_zip(archive_path, unsafe_name)

                with self.assertRaises(acquire_data.UnsafeArchiveError):
                    acquire_data.safe_extract_zip(archive_path, destination)
                self.assertFalse((root / "escape.txt").exists())

    def test_rejects_absolute_and_drive_paths(self) -> None:
        for index, unsafe_name in enumerate(("/absolute.txt", "C:\\absolute.txt")):
            with self.subTest(member=unsafe_name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                archive_path = root / f"absolute-{index}.zip"
                self._write_zip(archive_path, unsafe_name)

                with self.assertRaises(acquire_data.UnsafeArchiveError):
                    acquire_data.safe_extract_zip(archive_path, root / "extract")

    def test_rejects_symbolic_link_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "link.zip"
            link = zipfile.ZipInfo("link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(link, "../outside")

            with self.assertRaises(acquire_data.UnsafeArchiveError):
                acquire_data.safe_extract_zip(archive_path, root / "extract")

    def test_extracts_safe_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "safe.zip"
            self._write_zip(archive_path, "nested/data.csv", b"a,b\n1,2\n")

            extracted = acquire_data.safe_extract_zip(archive_path, root / "extract")

            self.assertEqual(len(extracted), 1)
            self.assertEqual(extracted[0].read_bytes(), b"a,b\n1,2\n")


class OfflineDownloadTests(unittest.TestCase):
    @staticmethod
    def _registry(source: Path, expected: str) -> dict:
        return {
            "schema_version": 1,
            "datasets": {
                "fixture": {
                    "name": "Offline fixture",
                    "description": "test",
                    "version": "1",
                    "provider": "tests",
                    "landing_page": "https://example.invalid/fixture",
                    "distribution_page": "https://example.invalid/fixture",
                    "access_type": "direct",
                    "credentials_required": False,
                    "license": {
                        "id": "test-license",
                        "url": "https://example.invalid/license",
                        "acknowledgement_required": False,
                    },
                    "files": [
                        {
                            "name": "fixture.csv",
                            "role": "fixture",
                            "url": source.as_uri(),
                            "archive_member": "fixture.csv",
                            "sha256": expected,
                        }
                    ],
                }
            },
        }

    def test_file_url_download_is_atomic_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            payload = b"id,label\n1,0\n"
            source.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            output = root / "raw"

            manifest_path = acquire_data.download_dataset(
                self._registry(source, expected), "fixture", output
            )

            installed = output / "fixture" / "fixture.csv"
            self.assertEqual(installed.read_bytes(), payload)
            self.assertFalse((installed.parent / "fixture.csv.part").exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["files"][0]["sha256"], expected)
            self.assertEqual(manifest["files"][0]["bytes"], len(payload))
            self.assertEqual(manifest["files"][0]["source_url"], source.as_uri())
            self.assertTrue(manifest["files"][0]["fetched_at"].endswith("Z"))

    def test_bad_download_hash_is_not_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            source.write_bytes(b"unexpected")
            output = root / "raw"

            with self.assertRaises(acquire_data.IntegrityError):
                acquire_data.download_dataset(
                    self._registry(source, "0" * 64), "fixture", output
                )

            destination = output / "fixture"
            self.assertFalse((destination / "fixture.csv").exists())
            self.assertFalse((destination / "fixture.csv.part").exists())


if __name__ == "__main__":
    unittest.main()
