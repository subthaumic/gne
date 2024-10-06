# TODO: Make the tests reference the appropriate distance functions instead
# of geometries (for legibility and logical reasons)

import pytest
import torch
from gne.utils.geometries import Euclidean, PoincareBall


def test_euclidean_distance():
    Eucl = Euclidean()

    # two simple tests that Euclidean distance function works correctly
    x = torch.tensor([1, 0])
    y = torch.tensor([0, 1])

    assert Eucl.distance(x, y) == pytest.approx(torch.sqrt(torch.tensor(2)))

    x = torch.tensor([0] * 4)
    y = torch.tensor([2] * 4)
    assert Eucl.distance(x, y) == 4

    # test that autograd works correctly for Euclidean distance function
    x = torch.tensor([1.0, 0, 0], requires_grad=True, dtype=torch.double)
    y = torch.tensor([0, 1.0, 0], requires_grad=True, dtype=torch.double)
    assert torch.autograd.gradcheck(Eucl.distance, [x, y], eps=1e-6, atol=1e-4)


def test_poincare_distance():
    Poin = PoincareBall()

    # two simple tests that Poincaré distance function works correctly
    x = torch.tensor([0.5, 0.0])
    y = torch.tensor([0.0, 0.5])
    assert Poin.distance(x, y) == pytest.approx(
        torch.acosh(torch.tensor(1 + 1 / 0.75**2))
    )

    x = torch.tensor([0, 0, 0, 0])
    y = torch.tensor([1, 0, 0, 0])
    assert torch.isinf(Poin.distance(x, y))

    # test that autograd works correctly for Poincare distance function
    x = torch.tensor([0.5, 0.0, 0.0], requires_grad=True, dtype=torch.double)
    y = torch.tensor([0.0, 0.5, 0.0], requires_grad=True, dtype=torch.double)
    assert torch.autograd.gradcheck(Poin.distance, [x, y], eps=1e-6, atol=1e-4)
