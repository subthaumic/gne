from gne.models.GeometricEmbedding import GeometricEmbedding, Config
from gne.utils.geometries import Euclidean

import torch


def test_training():
    x = torch.rand(16, 2)

    djinni = GeometricEmbedding(
        source_geometry=Euclidean(sample=x),
        target_geometry=Euclidean(dimension=2),
        config=Config(training={"epochs": 2}),
    )

    djinni(plot_loss=False)

    assert not djinni.target_geometry.sample.isnan().any()


# =============================================================================
# Code for experiments
# =============================================================================

# from gne.models.GeometricEmbedding import GeometricEmbedding, Config
# from gne.utils.geometries import Euclidean

# import torch

# x = torch.rand(16, 2)

# djinni = GeometricEmbedding(
#     source_geometry=Euclidean(sample=x),
#     config=Config(epochs=20, k_neighbours=3),
# )

# embedding = djinni(plot_loss=False)

# print(x)
# print(embedding)

# print(torch.norm(x - embedding, dim=1))
