from gne.models.GeometricEmbedding import GeometricEmbedding
from gne.models.Config import Config
from gne.utils.geometries import Euclidean

import torch
import sys

# Global Variables
source_dimension = 10
n_points = int(sys.argv[1])

# Create Data
torch.manual_seed(42)
points = torch.rand(n_points, source_dimension)

source = Euclidean(sample=points)

# set up gne
config = Config(epochs=100, k_neighbours=5)
djinni = GeometricEmbedding(source_geometry=source, config=config)

djinni()
