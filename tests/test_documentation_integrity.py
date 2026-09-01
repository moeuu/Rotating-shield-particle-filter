"""Tests for repository-owned documentation structure and local links."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
DOCS_PATH_PATTERN = re.compile(r"docs/[A-Za-z0-9_./-]+\.md")


def _local_markdown_targets(source_path: Path) -> tuple[Path, ...]:
    """Return resolved local targets referenced by one Markdown document."""
    content = source_path.read_text(encoding="utf-8")
    targets: list[Path] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(content):
        raw_target = match.group(1).strip().split(maxsplit=1)[0]
        target = raw_target.removeprefix("<").removesuffix(">")
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        path_part = target.split("#", maxsplit=1)[0]
        if path_part:
            targets.append((source_path.parent / path_part).resolve())
    return tuple(targets)


def test_all_local_markdown_links_resolve() -> None:
    """Every repository-local Markdown link must resolve to an existing path."""
    sources = (ROOT / "README.md", *sorted(DOCS_ROOT.rglob("*.md")))
    missing = sorted(
        str(target.relative_to(ROOT))
        for source in sources
        for target in _local_markdown_targets(source)
        if target.is_relative_to(ROOT) and not target.exists()
    )

    assert missing == []


def test_agent_document_references_resolve() -> None:
    """Every docs path named by AGENTS.md must identify a current document."""
    content = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    references = sorted(set(DOCS_PATH_PATTERN.findall(content)))

    assert references
    assert [reference for reference in references if not (ROOT / reference).is_file()] == []


def test_documentation_index_covers_every_current_document() -> None:
    """The documentation map must link every current document exactly once."""
    index_path = DOCS_ROOT / "README.md"
    indexed = {
        target
        for target in _local_markdown_targets(index_path)
        if target.suffix == ".md" and target != index_path.resolve()
    }
    current = {
        path.resolve()
        for path in DOCS_ROOT.rglob("*.md")
        if path != index_path
    }

    assert indexed == current
    assert not (DOCS_ROOT / "archive").exists()
