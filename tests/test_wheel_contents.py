"""Check source-only wheels and source distributions in disposable workspaces."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("baselines", "evaluation", "pf", "planning", "visualization")


def _build(project: Path, output: Path, kind: str) -> Path:
    """Build one distribution without installing its runtime dependencies."""
    subprocess.run(
        ["uv", "build", f"--{kind}", "--out-dir", str(output)],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    pattern = "*.whl" if kind == "wheel" else "*.tar.gz"
    artifacts = tuple(output.glob(pattern))
    assert len(artifacts) == 1
    return artifacts[0]


def _assert_wheel_sources(wheel: Path, sources: dict[str, bytes]) -> None:
    """Require exact module contents and only distribution metadata otherwise."""
    with zipfile.ZipFile(wheel) as archive:
        files = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    modules = {n: value for n, value in files.items() if n.endswith(".py")}
    assert modules.keys() == sources.keys()
    assert modules == sources
    assert all(n in sources or ".dist-info/" in n for n in files)
    entry_points = next(v for n, v in files.items() if n.endswith("/entry_points.txt"))
    assert b"rotating-shield-pf-live = pf.closed_loop:main" in entry_points
    assert b"rotating-shield-pf =" not in entry_points


def test_distributions_ignore_stale_build_trees_and_match_current_source(
    tmp_path: Path,
) -> None:
    """Old build modules must never enter direct or sdist-rebuilt wheels."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", ".gitignore"):
        shutil.copy2(ROOT / name, project / name)
    shutil.copytree(
        ROOT / "src", project / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    sources = {
        p.relative_to(project / "src").as_posix(): p.read_bytes()
        for package in PACKAGES
        for p in (project / "src" / package).rglob("*.py")
    }
    for relative in (
        "build/lib/pf/retired_marker.py",
        "build/lib/planning/legacy_program_guard.py",
        "results/private-truth.json",
        "logs/run/console.log",
        "data/manchester_nuclear_assets/usd/scene.usda",
        "tmp/preview.png",
        "src/pf/__pycache__/retired_marker.cpython-312.pyc",
        "src/obsolete.egg-info/SOURCES.txt",
    ):
        stale = project / relative
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"must not be packaged\n")
    # Existing package files in an old build must not replace current content.
    (project / "build/lib/pf/__init__.py").write_bytes(b"raise RuntimeError('old')\n")

    wheel = _build(project, tmp_path / "direct", "wheel")
    _assert_wheel_sources(wheel, sources)
    sdist = _build(project, tmp_path / "source-dist", "sdist")
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        forbidden = {"build", "results", "logs", "data", "tmp", "__pycache__"}
        assert all(not (set(Path(m.name).parts) & forbidden) for m in members)
        assert all(not any(p.endswith(".egg-info") for p in Path(m.name).parts)
                   for m in members)
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = next((tmp_path / "unpacked").iterdir())
    rebuilt = _build(unpacked, tmp_path / "rebuilt", "wheel")
    _assert_wheel_sources(rebuilt, sources)

    # Import the installed distribution, not the editable checkout under test.
    installed = tmp_path / "installed"
    subprocess.run(
        ["uv", "pip", "install", "--no-deps", "--target", str(installed), str(wheel)],
        check=True, capture_output=True, text=True, timeout=120.0,
    )
    subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
         "import pf.closed_loop, evaluation.completed_run; "
         "assert Path(pf.closed_loop.__file__).is_relative_to(sys.argv[1])",
         str(installed)],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60.0,
    )
