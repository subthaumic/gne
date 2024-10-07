
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# gNE

Geometric Neighbour Embeddings (gNE) is a dimensionality reduction technique similar to t-SNE and UMAP, designed to embed high-dimensional data into a low-dimensional space while preserving the local geometric relationships between data points.

gNE is a work-in-progress experiment to try to improve interpretability of features in the low-dimensional space directly from geometric properties like distances, areas, volumes, and does not use affinities.

The gNE algorithm operates under the following principles
- data are drawn from a high-dimensional Riemannian manifold *(source geometry)*
- the target embedding is a lower-dimensional Riemannian manifold *(target geometry)*
- similar data points in the high-dimensional space and their higher-order neighbourhood structure in target geometry should remain close to the original source.

To achieve the last point, the optimization factors through a comparison of simplicial complexes (default: $k$-nearest neighbour complex) that capture the higher-order neighbourhood structure of the data in the source and target geometry.


## Installation

gNE is not yet available through PyPI/pip as it is still early work.

To install the package, you need to clone this repository and install it locally.

```bash
git clone https://github.com/subthaumic/gne.git
cd gne
pip install .
```

Requirements:
- Python 3.7 or greater
- torch
- numpy
- scipy
- scikit-learn

For visualization purposes, you may also wish to install:
- matplotlib
- seaborn

## Basic Usage

The gNE package provides an easy-to-use interface inspired by PyTorch-like APIs, allowing users to obtain embeddings by directly calling the model instance.

Here is a simple example of how to use gNE for dimensionality reduction:

```python
import torch
from gne.models.GeometricEmbedding import GeometricEmbedding
from gne.utils.geometries import Euclidean

# Create synthetic data
points = torch.rand(20, 10)  # 20 points, 10-dimensional data

# Instantiate GeometricEmbedding with default settings
djinni = GeometricEmbedding(source_geometry=Euclidean(sample=points))

# Run the optimization and get the low-dimensional embedding
embedding = djinni()
print(embedding)
```

## Configuration

The `Config` class in gNE handles configuration settings in a flexible manner, grouping attributes into a dictionary of dictionaries that can be updated both during instantiation and at runtime.
This allows for convenient adjustments of key settings like `source_geometry`, `target_geometry`, and `optimizer` parameters.

The `Config` class reads default values from `config.yaml` and provides an easy way to override these through keyword arguments during initialization.
It also supports dynamic attribute access using python's dot notation for ease of use.

Attributes managed by `Config` are:
- **source_geometry**: Settings related to the source geometry in which the input data lives (default: Euclidean).
- **target_geometry**: Settings for the target embedding geometry (default: Euclidean, 2 dimensions, PCA initialization).
- **source_complex**: Parameters defining the initial complex calculated from the input data (default: k-nearest neighbour, k=5).
- **target_complex**: Configuration related to the target complex.
- **loss**: Settings for the objective function used for optimization (default: $L^2$).
- **training**: Optimization parameters like batch size and learning rate.
- **scheduler**: Scheduler configuration for learning rate adjustments.
- **earlystop**: Criteria for early stopping optimization.
- **output**: Settings for output paths, formats, and verbosity.


```python
from gne import Config

config = Config(
    source_geometry={'dimension': 3},
    target_geometry={'initialization_method': 'random', 'dimension': 2},
    source_complex={'k_neighbours': 5, 'max_dim': 2},
    target_complex={'k_neighbours': 4, 'max_dim': 2},
    loss={'lagrange_multipliers': [1.0, 0.5]},
    training={'epochs': 50, 'batch_size': 32, 'learning_rate': 0.01, 'burnin': 5},
    scheduler={'factor': 0.5, 'patience': 10, 'cooldown': 2},
    earlystop={'quantile': 0.9, 'lr_threshold': 0.001, 'patience': 5},
    output={'interim_data_path': './data/', 'plot_loss': True}
)
```



## Performance

gNE is *not* designed to handle large datasets efficiently.
It's focus on higher-order structures currently makes it slow and parallelization difficult (not implemented).

## Examples

See the 'notebooks' directory for first examples of how to use gNE for dimensionality reduction and visualization.

## Citation

If you use geometric Neighbour Embeddings (gNE) in your work, please cite the repository:

```bibtex
@misc{Bleher2024gne,
  author = {Michael Bleher},
  title = {Geometric Neighbour Embeddings (gNE)},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/subthaumic/gNE}}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
Any improvements, including code, documentation, or examples, are highly appreciated.
