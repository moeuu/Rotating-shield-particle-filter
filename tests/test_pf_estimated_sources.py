"""Tests for exporting estimated sources from the PF."""

import numpy as np

from pf.parallel import ParallelIsotopePF, EstimatedSource
from pf.particle_filter import PFConfig, IsotopeParticle
from pf.state import IsotopeState


def test_get_estimated_sources_returns_dataclasses() -> None:
    """Estimated sources should be returned as dataclasses with position/strength."""
    config = PFConfig(num_particles=1)
    pf = ParallelIsotopePF(isotope_names=["Cs-137"], config=config)
    filt = pf.filters["Cs-137"]
    # inject a simple continuous particle with one source
    filt.continuous_particles = [
        IsotopeParticle(
            state=IsotopeState(num_sources=1, positions=np.array([[0.1, 0.2, 0.3]]), strengths=np.array([5.0]), background=0.0),
            log_weight=0.0,
        )
    ]
    sources = pf.get_estimated_sources()
    assert "Cs-137" in sources
    assert len(sources["Cs-137"]) == 1
    src = sources["Cs-137"][0]
    assert isinstance(src, EstimatedSource)
    assert np.allclose(src.position, [0.1, 0.2, 0.3])
    assert src.strength == 5.0
