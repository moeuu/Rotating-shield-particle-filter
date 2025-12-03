"""チューニング関連のヘルパーの動作確認。"""

import numpy as np

from spectrum.tuning import get_best_parameters, simulate_reference_spectrum, evaluate_spectrum_quality


def test_get_best_parameters_runs():
    """グリッドサーチが実行できベストパラメータを返すことを確認。"""
    c, b, bg, q = get_best_parameters()
    assert c > 0
    assert b >= 0
    assert bg >= 0
    assert isinstance(q.passes, bool)


def test_reference_spectrum_quality_with_best_params():
    """ベストパラメータで生成したスペクトルが品質判定を返す。"""
    c, b, bg, _ = get_best_parameters()
    E, spec = simulate_reference_spectrum(continuum_to_peak=c, backscatter_fraction=b, background_rate_cps=bg, rng_seed=1)
    q = evaluate_spectrum_quality(E, spec)
    assert isinstance(q.passes, bool)
