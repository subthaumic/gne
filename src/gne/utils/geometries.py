from gne.utils.distances import (
    Distance,
    EuclideanDistance,
    GeodesicDistance,
    PoincareDistance,
)
from gne.utils.metrics import RiemannianMetric, EuclideanMetric, PoincareMetric

import torch
import itertools


class Geometry:
    """
    A base class representing a geometric space, defined by a Riemannian metric (or
    a distance function). If only the Riemannian metric is provided, will use an
    approximation of associated geodesic distance.

    Parameters:
    - metric (RiemannianMetric, optional): The metric that defines the geometry.
    - distance (Distance, optional): The distance function used to measure distances.
    - sample (torch.Tensor, optional): A sample of points from the space.
    - sample_size (int, optional): The number of points in the sample.
    - dimension (int, optional): The dimensionality of the space.
    """

    def __init__(
        self,
        metric: RiemannianMetric = None,
        distance: Distance = None,
        sample: torch.Tensor = None,
        sample_size: int = None,
        dimension: int = None,
    ):
        self.metric = metric

        self.sample_size = sample_size
        self.dimension = dimension

        # Initialize the distance function based on provided metric or distance.
        if distance is not None:
            self.distance = distance
        elif self.metric is not None:
            self.distance = GeodesicDistance(self.metric)
        else:
            raise ValueError("Either metric or distance function must be provided.")

        # Initialize sample points near zero
        if sample is not None:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            self.sample = sample
        elif (self.sample_size is not None) and (self.dimension is not None):
            self.sample = torch.rand(
                sample_size, dimension, dtype=torch.float64
            ).uniform_(-1e-1, 1e-1)
        else:
            self.sample = None

        # Update sample size and dimension based on the actual sample.
        if self.sample is not None:
            self.sample_size = self.sample.size()[0]
            self.dimension = self.sample.size()[1]

    def __repr__(self):
        out = "gne.Geometry("
        if self.metric is not None:
            out += f"metric: {self.metric}, "
        if self.distance is not None:
            out += f"distance: {self.distance}, "
        if self.sample_size is not None:
            out += f"sample_size: {self.sample_size}, "
        if self.dimension is not None:
            out += f"dimension: {self.dimension}"
        out += ")"
        return out

    def __call__(self, indices):
        """
        Returns a new Geometry instance with a subset of the original sample points.
        """
        return Geometry(
            metric=self.metric,
            distance=self.distance,
            sample=self.sample[indices],
            dimension=self.dimension,
            sample_size=len(indices),
        )

    def add_sample(self, new_points):
        """
        Adds new points to the existing sample points.
        """
        if not isinstance(new_points, torch.Tensor):
            new_points = torch.tensor(new_points)

        if self.sample is None:
            self.sample = new_points
        else:
            self.sample = torch.cat((self.sample, new_points))

        self.sample_size = self.sample.size()[0]
        self.dimension = self.sample.size()[1]

    def compute_distances(self, indices=None):
        """
        Computes pairwise distances between sample points using the defined distance
        function.

        Parameters:
        - indices (optional): Specifies a subset of sample points to consider.
            If None, uses all points.

        Returns:
        - A matrix of pairwise distances.
        """
        if indices is None:
            indices = torch.arange(self.sample_size)
        dists = torch.zeros((len(indices), len(indices)))
        for (i, a), (j, b) in itertools.combinations(enumerate(indices), 2):
            dists[j, i] = dists[i, j] = self.distance(self.sample[a], self.sample[b])
        return dists


class Euclidean(Geometry):
    """
    Represents Euclidean space, a subclass of Geometry.

    Parameters:
    - sample (torch.Tensor, optional): A sample of points from Euclidean space.
    - sample_size (int, optional): The number of points in the sample.
    - dimension (int, optional): The dimensionality of the Euclidean space.
    """

    def __init__(
        self,
        sample: torch.Tensor = None,
        sample_size: int = None,
        dimension: int = None,
    ):
        super().__init__(
            distance=EuclideanDistance(),
            metric=EuclideanMetric(),
            sample=sample,
            sample_size=sample_size,
            dimension=dimension,
        )

    def compute_distances(self, indices=None):
        """
        Computes pairwise distances between sample points, utilizing scipy's
        spatial distance functions for efficiency

        Parameters:
        - indices (optional): subset of sample points to consider.
            If None, uses all points.

        Returns:
        - A matrix of pairwise distances.
        """
        if indices is None:
            indices = torch.arange(self.sample_size)
        dists = torch.nn.functional.pdist(self.sample[indices], p=2)
        # TODO: make consistente between upper triangular and squareform
        # currenetly can't use squareform, because that needs detach
        # dists = torch.Tensor(squareform(dists.detach()))
        return dists


class PoincareBall(Geometry):
    """
    Represents the Poincaré ball model of hyperbolic space, a subclass of Geometry.

    Parameters:
    - sample (torch.Tensor, optional): A sample of points from the Poincaré ball.
    - sample_size (int, optional): The number of points in the sample.
    - dimension (int, optional): The dimensionality of the Poincaré ball.
    """

    def __init__(
        self,
        sample: torch.Tensor = None,
        sample_size: int = None,
        dimension: int = None,
    ):
        super().__init__(
            distance=PoincareDistance(),
            metric=PoincareMetric(),
            sample=sample,
            sample_size=sample_size,
            dimension=dimension,
        )
        self.max_norm = 1  # The maximum norm for points in the Poincaré ball space

        # TODO: make sure sample is contained in Poincaré ball: check norm <= max_norm

    # TODO: Override `compute_distances` with an optimized implementation for Poincaré
    # distances, similar to the Euclidean class.
