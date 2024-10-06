from sklearn.neighbors import KDTree
import itertools

import torch

from gne.utils.geometries import Geometry
from gne.utils.complex import Simplex, Complex


def geometric_weights(geometry):
    match geometry.sample.size(0):
        # TODO: UserWarning "geometric weights only implemented for Euclidean space"
        case 2:
            distance = geometry.distance(geometry.sample[0], geometry.sample[1])
            return distance
        case 3:
            # calculate Euclidean area from distances using Heron's formula
            a = geometry.distance(geometry.sample[0], geometry.sample[1])
            b = geometry.distance(geometry.sample[1], geometry.sample[2])
            c = geometry.distance(geometry.sample[0], geometry.sample[2])
            s = (a + b + c) / 2  # Semiperimeter
            radicand = torch.clamp(
                s * (s - a) * (s - b) * (s - c), min=1e-9
            )  # Ensure positive and valid autograd
            area = radicand.sqrt()
            return area
        case _:
            # TODO: UserWarning "Volume for high dimensional simplices not implemented"
            return None


def topological_weights(geometry):
    # TODO: implement this correctly.
    # should return 1 if simplex is present and None (or 0?) otherwise.
    return 1


def distance_weights(geometry):
    match geometry.sample.size(0):
        case 2:
            distance = geometry.distance(geometry.sample[0], geometry.sample[1])
            return distance
        case _:
            return None


# def cauchy_weights(geometry):
#     match geometry.sample.size(0):
#         case 2:
#             distance = geometry.distance(geometry.sample[0], geometry.sample[1])
#             return 1 / (1 + distance**2)
#         case 3:
#             a = geometry.distance(geometry.sample[0], geometry.sample[1])
#             b = geometry.distance(geometry.sample[1], geometry.sample[2])
#             c = geometry.distance(geometry.sample[0], geometry.sample[2])
#             s = (a + b + c) / 2  # Semiperimeter
#             area = (s * (s - a) * (s - b) * (s - c)).sqrt()  # Heron's formula
#             return 1 / (1 + area**2)
#         case _:
#             # TODO: UserWarning "Volume for higher
#                               dimensional simplices not implemented"
#             return None


def kneighbors_complex(
    geometry: Geometry,
    k_neighbours: int,
    max_dim: int = None,
    weight_fn=geometric_weights,
):
    if max_dim is None:
        max_dim = k_neighbours - 1
    # determine k nearest neighbour classes
    points = geometry.sample.detach()
    kdt = KDTree(points, leaf_size=30, metric="euclidean")
    neighbourhoods = kdt.query(points, k=k_neighbours, return_distance=False)
    # create complex
    complex = Complex()
    for neighbours in neighbourhoods:
        for ids in itertools.combinations(neighbours, max_dim + 1):
            complex.add_simplex(Simplex(ids))

    # fill in weights
    if weight_fn is not None:
        for simplex in complex:
            indices = [int(v) for v in simplex.vertices]
            weight = weight_fn(geometry(indices))
            complex.update_weight(Simplex(indices), weight)

    return complex
