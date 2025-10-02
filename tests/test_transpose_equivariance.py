import numpy as np

from ccf_key_detector.ske import SKEBackend, SKEConfig, create_ske
from ccf_key_detector.transpose import (
    TranspositionConfig,
    create_operator,
    scan_distribution,
)


def test_transposition_matrix_matches_roll() -> None:
    operator = create_operator(TranspositionConfig(n_bins=5))
    vector = [1, 2, 3, 4, 5]
    assert operator.apply(vector, 1) == [2, 3, 4, 5, 1]
    assert operator.apply(vector, -1) == [5, 1, 2, 3, 4]
    matrix = np.asarray(operator.matrix())
    expected = np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(matrix, expected)


def test_transposition_equivariance_of_scan() -> None:
    n_bins = 24
    operator = create_operator(TranspositionConfig(n_bins=n_bins))
    ske = create_ske(SKEConfig(mode=SKEBackend.KK_MAJOR))

    rng = np.random.default_rng(0)
    base_vector = rng.random(n_bins)
    base_vector /= base_vector.sum()

    base_distribution = scan_distribution(base_vector, ske, operator)

    shift = 5
    shifted_vector = np.array(operator.apply(base_vector, shift))
    shifted_distribution = scan_distribution(shifted_vector, ske, operator)

    np.testing.assert_allclose(
        shifted_distribution,
        np.roll(base_distribution, -shift),
        atol=1e-8,
    )
