"""Tests for RA-L manuscript policy-critical LaTeX content."""

from __future__ import annotations

from scripts.ral_figure_common import LATEX_ROOT


def test_ral_funding_acknowledgment_is_first_page_footnote() -> None:
    """The anonymous RA-L funding acknowledgment must stay in the author footnote."""
    main_tex = LATEX_ROOT / "main.tex"
    content = main_tex.read_text(encoding="utf-8")

    assert r"\IEEEoverridecommandlockouts" in content
    assert (
        r"\newcommand{\RALFundingAcknowledgment}"
        "{This work was in part supported by XXX.}"
    ) in content
    assert r"\thanks{\RALFundingAcknowledgment}" in content
    assert r"\section*{ACKNOWLEDGMENT}" not in content
