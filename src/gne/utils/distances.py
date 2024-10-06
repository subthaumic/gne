import torch

from gne.utils.metrics import RiemannianMetric
from gne.utils.numerics import geodesic


class Distance(torch.nn.Module):
    def __init__(self):
        super().__init__()


class EuclideanDistance(Distance):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y):
        return torch.sqrt(torch.sum((x - y) ** 2, dim=-1))


class PoincareDistance(Distance):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(u, v):
        delta = (
            2
            * torch.sum((u - v) ** 2, dim=-1)
            / ((1 - torch.sum(u**2, dim=-1)) * (1 - torch.sum(v**2, dim=-1)))
        )
        return torch.acosh(1 + delta)


class GeodesicDistance(Distance):
    def __init__(self, metric: RiemannianMetric):
        super().__init__()
        self.metric = metric

    def forward(self, x, y):
        _, length = geodesic(x, y, self.metric)
        return length
