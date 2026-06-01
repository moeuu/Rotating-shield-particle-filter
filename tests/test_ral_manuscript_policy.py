"""Tests for RA-L manuscript policy-critical LaTeX content."""

from __future__ import annotations

from pathlib import Path


RAL_MANUSCRIPT_ROOT = (
    Path(__file__).resolve().parents[2] / "latex" / "projects" / "ieee-ra-l-letter"
)


def test_ral_funding_acknowledgment_is_first_page_footnote() -> None:
    """The anonymous RA-L funding acknowledgment must stay in the author footnote."""
    main_tex = RAL_MANUSCRIPT_ROOT / "main.tex"
    content = main_tex.read_text(encoding="utf-8")

    assert r"\IEEEoverridecommandlockouts" in content
    assert (
        r"\newcommand{\RALFundingAcknowledgment}"
        "{This work was in part supported by XXX.}"
    ) in content
    assert r"\thanks{\RALFundingAcknowledgment}" in content
    assert r"\section*{ACKNOWLEDGMENT}" not in content

