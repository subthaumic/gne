import torch

from gne.utils.geometries import Euclidean
from gne.utils.complex import Simplex, Complex
from gne.utils.geometry_to_complex import kneighbors_complex, geometric_weights


def test_kneighbors_complex_with_weight_fn_none():
    geometry = Euclidean(
        torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float)
    )
    k_neighbours = 2

    expected_complex = Complex()
    expected_complex.add_simplex(Simplex([0, 1], None))
    expected_complex.add_simplex(Simplex([1, 2], None))
    expected_complex.add_simplex(Simplex([2, 3], None))

    result_complex = kneighbors_complex(geometry, k_neighbours, weight_fn=None)

    assert result_complex == expected_complex

    for simplex in result_complex:
        assert simplex.weight == expected_complex.get_weight(simplex)


def test_kneighbors_complex_with_geometric_weights():
    geometry = Euclidean(
        torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float)
    )
    k_neighbours = 2
    max_dim = 1

    expected_complex = Complex()
    expected_complex.add_simplex(Simplex([0, 1], geometric_weights(geometry([0, 1]))))
    expected_complex.add_simplex(Simplex([1, 2], geometric_weights(geometry([1, 2]))))
    expected_complex.add_simplex(Simplex([2, 3], geometric_weights(geometry([2, 3]))))

    result_complex = kneighbors_complex(
        geometry, k_neighbours, max_dim, weight_fn=geometric_weights
    )

    assert result_complex == expected_complex

    for simplex in result_complex:
        assert simplex.weight == expected_complex.get_weight(simplex)


def test_kneighbors_complex_with_custom_weight_fn():
    def custom_weight_fn(geometry):
        match geometry.sample.size(0):
            case 2:
                return 0.5
            case _:
                return None

    geometry = Euclidean(
        torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float)
    )
    k_neighbours = 2
    max_dim = 1

    expected_complex = Complex()
    expected_complex.add_simplex(Simplex([0, 1], custom_weight_fn(geometry([0, 1]))))
    expected_complex.add_simplex(Simplex([1, 2], custom_weight_fn(geometry([1, 2]))))
    expected_complex.add_simplex(Simplex([2, 3], custom_weight_fn(geometry([2, 3]))))

    result_complex = kneighbors_complex(
        geometry, k_neighbours, max_dim, weight_fn=custom_weight_fn
    )

    assert result_complex == expected_complex

    for simplex in result_complex:
        assert simplex.weight == expected_complex.get_weight(simplex)


def test_kneighbors_complex_with_max_dim_none():
    geometry = Euclidean(
        torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float)
    )
    k_neighbours = 4
    max_dim = None

    expected_complex = Complex()
    expected_complex.add_simplex(Simplex([0, 1], geometric_weights(geometry([0, 1]))))
    expected_complex.add_simplex(Simplex([0, 2], geometric_weights(geometry([0, 2]))))
    expected_complex.add_simplex(Simplex([0, 3], geometric_weights(geometry([0, 3]))))
    expected_complex.add_simplex(Simplex([1, 2], geometric_weights(geometry([1, 2]))))
    expected_complex.add_simplex(Simplex([1, 3], geometric_weights(geometry([1, 3]))))
    expected_complex.add_simplex(Simplex([2, 3], geometric_weights(geometry([2, 3]))))
    expected_complex.add_simplex(
        Simplex([0, 1, 2], geometric_weights(geometry([0, 1, 2])))
    )
    expected_complex.add_simplex(
        Simplex([0, 1, 3], geometric_weights(geometry([0, 1, 3])))
    )
    expected_complex.add_simplex(
        Simplex([0, 2, 3], geometric_weights(geometry([0, 2, 3])))
    )
    expected_complex.add_simplex(
        Simplex([1, 2, 3], geometric_weights(geometry([1, 2, 3])))
    )
    expected_complex.add_simplex(
        Simplex([0, 1, 2, 3], geometric_weights(geometry([0, 1, 2, 3])))
    )

    result_complex = kneighbors_complex(
        geometry, k_neighbours, max_dim, weight_fn=geometric_weights
    )

    assert result_complex == expected_complex

    for simplex in result_complex:
        assert simplex.weight == expected_complex.get_weight(simplex)
