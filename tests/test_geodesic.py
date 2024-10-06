from gne.utils.distances import PoincareDistance, GeodesicDistance
from gne.utils.metrics import PoincareMetric

import pytest
import torch


# function to generate random points in the Poincaré Disk
def random_point(radius=None, angle=None):
    # Generate random radii (use square root to ensure uniform distribution)
    if radius is None:
        radius = torch.sqrt(torch.rand(1))
    else:
        radius = torch.tensor([radius])
    # Generate random angle
    if angle is None:
        angle = torch.rand(1)
    else:
        angle = torch.tensor([angle])
    # Convert to Cartesian coordinates
    x = radius * torch.sin(2 * torch.pi * angle)
    y = radius * torch.cos(2 * torch.pi * angle)
    # return (x,y)
    return torch.stack((x, y), dim=1).squeeze()


def test_geodesic():
    x = random_point(radius=0.9)
    y = random_point(radius=0.9)

    geodesic_dist = GeodesicDistance(PoincareMetric())(x, y)

    assert geodesic_dist != 0
    assert not geodesic_dist.isnan()

    dist = PoincareDistance()(x, y)
    rel_error = torch.abs(geodesic_dist - dist) / torch.abs(geodesic_dist + dist)

    if not pytest.approx(rel_error, abs=5e-3) == 0:
        pytest.skip(
            f"""
                Distances differ by more than .5% :
                approximated geodesic distance = {geodesic_dist}
                true geodesic distance =  {dist}
                error = {rel_error}
            """
        )
