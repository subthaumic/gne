import torch
from tqdm import tqdm
import numpy as np

from gne.utils.metrics import RiemannianMetric


def geodesic(
    x: torch.Tensor(),
    y: torch.Tensor(),
    metric: RiemannianMetric(),
    max_density: int = 10,
):
    ## Optimization settings for geodesic (hard coded!)
    num_epochs = 1000
    pbar = tqdm(range(1, num_epochs + 1), ncols=100, leave=False)
    earlystop_count = 0
    epoch_loss = []
    best_lengths = []

    # detach x and y from potential superordinate autograd computational graph
    start_point = x.detach()
    end_point = y.detach()

    # use Euclidean geodesic as an initial guess
    geodesic = torch.cat(
        [
            start_point[None, :],
            0.5 * start_point[None, :] + 0.5 * end_point[None, :],
            end_point[None, :],
        ],
        dim=0,
    )

    # prepare interior points for optimization
    interior_points = geodesic[1:-1].requires_grad_()
    optimizer = torch.optim.Adam([interior_points], lr=0.01)

    # update geodesic to be dependent on interior_points (necessary for autograd)
    geodesic = torch.cat(
        [start_point[None, :], interior_points, end_point[None, :]], dim=0
    )
    # line element for discrete differentiation and integration
    dt = 1 / (geodesic.size(0) - 1)

    for epoch in pbar:
        optimizer.zero_grad()

        # use Euclidean difference quotient as proxy
        # for tangent vectors along the path
        tangents = (geodesic[1:] - geodesic[:-1]) / dt

        # calculate norm of proxy tangent vectors w.r.t. metric
        # at the two possible base_points
        tangent_norms_fwd = torch.sqrt(
            torch.stack(
                [
                    metric(tangents[i], tangents[i], base_point=geodesic[i])
                    for i in range(tangents.size(0))
                ]
            )
        )
        tangent_norms_bwd = torch.sqrt(
            torch.stack(
                [
                    metric(tangents[i], tangents[i], base_point=geodesic[i + 1])
                    for i in range(tangents.size(0))
                ]
            )
        )

        # calculate length, symmetrized over forward and backward calculation
        length = torch.sum(tangent_norms_fwd + tangent_norms_bwd) * dt / 2
        epoch_loss.append(length)

        # backwards pass and optimization
        length.backward()
        _ = torch.nn.utils.clip_grad_norm_(interior_points, max_norm=100.0)
        optimizer.step()

        # update geodesic (backwards pass only changes interior_points)
        geodesic = torch.cat(
            [start_point[None, :], interior_points, end_point[None, :]], dim=0
        )

        # update progress bar
        pbar.set_description(f"length: {str(np.round(length.item(),2))}")

        # check whether at current resolution length still changes more
        # than 1% per iteration
        if len(epoch_loss) >= 2:
            if abs(epoch_loss[-1] - epoch_loss[-2]) < 1.0e-2:
                # save current length as best guess
                best_lengths.append(length)

                # subdivide geodesic as long as number of points per unit length
                # is less than max_density
                if geodesic.size(0) / length < max_density:
                    # detach geodesic from autograd for new
                    # optimization in next epoch
                    geodesic = geodesic.detach()

                    # subdivide all current segments of geodesic
                    # into two segments each
                    midpoints = (geodesic[:-1] + geodesic[1:]) / 2
                    subdivision = torch.empty(
                        (geodesic.size(0) + midpoints.size(0), geodesic.size(1))
                    )
                    subdivision[0::2] = geodesic  # Original points in even indices
                    subdivision[1::2] = midpoints  # Midpoints in odd indices

                    # prepare optimization of interior_points
                    interior_points = subdivision[1:-1].requires_grad_()
                    optimizer = torch.optim.Adam([interior_points], lr=0.01)

                    # update geodesic and line element
                    geodesic = torch.cat(
                        [start_point[None, :], interior_points, end_point[None, :]],
                        dim=0,
                    )
                    dt = 1 / (geodesic.size(0) - 1)

        # stop if best length stagnates or subdivisions don't improve the result
        if 2 <= len(best_lengths):
            if abs(best_lengths[-1] - best_lengths[-2]) < 1.0e-4:
                earlystop_count += 1
        if earlystop_count > 10:
            # print(f"\nStopped at epoch {epoch}")
            break

    pbar.close()

    return geodesic.detach(), length.detach()
