from gne.utils.distances import EuclideanDistance, GeodesicDistance
from gne.utils.metrics import EuclideanMetric, SasakiMetric

import pytest
import torch


def test_sasaki():
    g = SasakiMetric(EuclideanMetric())
    dist = GeodesicDistance(g)

    true_dist = EuclideanDistance()

    x, y = torch.rand(2, 4)
    assert pytest.approx(dist(x, y), rel=1e-3) == true_dist(x, y)
