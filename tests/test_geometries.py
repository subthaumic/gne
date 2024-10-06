from gne.utils.distances import EuclideanDistance, GeodesicDistance, PoincareDistance
from gne.utils.metrics import EuclideanMetric, PoincareMetric
from gne.utils.geometries import Geometry, Euclidean, PoincareBall

import pytest
import torch


@pytest.fixture
def sample_data():
    x = (
        torch.rand(10, 4) * 2 - 1
    )  # 10 points in 4D space, uniformly distributed [-1,1]^4
    max_norm = torch.max(torch.norm(x, p=2, dim=1))
    # rescale if points are not contained in unit ball for use with Poincaré Distances
    if max_norm > 1:
        x = (1 - 1e-3) / max_norm * x
    return x


# =============================================================================
# Tests for Geometry Class
# =============================================================================


def test_init_with_metric():
    metric = EuclideanMetric()
    geom = Geometry(metric=metric)
    assert geom.metric == metric
    assert isinstance(geom.distance, GeodesicDistance)


def test_init_with_distance():
    distance = EuclideanDistance()
    geom = Geometry(distance=distance)
    assert geom.distance == distance


def test_call_returns_new_instance(sample_data):
    geom = Geometry(metric=EuclideanMetric(), sample=sample_data)
    subset_geom = geom([0, 1, 2])
    assert isinstance(subset_geom, Geometry)
    assert subset_geom.sample_size == 3


def test_add_sample():
    geom = Geometry(metric=EuclideanMetric())
    new_points = torch.rand(5, 4) * 2 - 1
    geom.add_sample(new_points)
    assert geom.sample_size == 5


def test_add_sample_to_existing(sample_data):
    geom = Geometry(metric=EuclideanMetric(), sample=sample_data)
    new_points = torch.rand(5, 4) * 2 - 1
    geom.add_sample(new_points)
    assert geom.sample_size == 15


def test_compute_distances(sample_data):
    geom = Geometry(sample=sample_data, distance=EuclideanDistance())
    distances = geom.compute_distances()
    assert distances.shape == (10, 10)
    assert distances.isfinite().all()


# =============================================================================
# Tests for Euclidean Class
# =============================================================================


def test_euclidean_init(sample_data):
    Eucl = Euclidean(sample=sample_data)
    assert isinstance(Eucl.distance, EuclideanDistance)
    assert isinstance(Eucl.metric, EuclideanMetric)


def test_euclidean_compute_distances(sample_data):
    Eucl = Euclidean(sample=sample_data)
    distances = Eucl.compute_distances()
    print(sample_data)
    print(distances)
    assert distances.shape.numel() == len(sample_data) * (len(sample_data) - 1) / 2
    assert distances.isfinite().all()


# =============================================================================
# Tests for PoincareBall Class
# =============================================================================


def test_poincareball_init(sample_data):
    Poinc = PoincareBall(sample=sample_data)
    assert isinstance(Poinc.distance, PoincareDistance)
    assert isinstance(Poinc.metric, PoincareMetric)


# Placeholder for 'compute_distances' test
def test_poincareball_compute_distances(sample_data):
    Poinc = PoincareBall(sample=sample_data)
    distances = Poinc.compute_distances()
    assert distances.shape == (10, 10)
    assert distances.isfinite().all()
