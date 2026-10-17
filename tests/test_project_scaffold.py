from pathlib import Path


def test_project_uses_native_neon_workflow_without_docker() -> None:
    """The deployment contract is hosted Neon plus a native Python process."""
    assert not Path("Dockerfile").exists()
    assert not Path("docker-compose.yml").exists()
    assert not Path("docker-compose.yaml").exists()


def test_required_sprint_directories_exist() -> None:
    for directory in (
        "data", "notebooks", "backend", "frontend", "docs", "monitoring"
    ):
        assert Path(directory).is_dir(), directory
