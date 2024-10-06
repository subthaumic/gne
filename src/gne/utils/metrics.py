import torch


class RiemannianMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_metric_tensor(self, base_points):
        """
        Computes the matrix representation of the metric for a batch of base_points on
        the manifold.
        """
        batch_size, dim = base_points.shape

        # Create batch of identity matrices
        identity_matrices = (
            torch.eye(dim, device=base_points.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Compute the metric tensor for each base point
        g = torch.zeros(
            (batch_size, dim, dim), dtype=torch.float64, device=base_points.device
        )

        for i in range(dim):
            for j in range(dim):
                g[:, i, j] = self.forward(
                    identity_matrices[:, i], identity_matrices[:, j], base_points
                )

        return g

    def get_christoffels(self, base_points):
        """
        Computes the Christoffel symbols for a batch of base_points on the manifold.
        """
        batch_size, dim = base_points.shape

        # Create batch of identity matrices
        (
            torch.eye(dim, device=base_points.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # get batch of metric tensors
        g = self.get_metric_tensor(base_points)

        # Compute inverses of the batch of metric tensors
        g_inv = torch.linalg.inv(g)

        # Compute the partial derivatives of the metric tensor components
        dg_dx = torch.zeros(
            (batch_size, dim, dim, dim), dtype=torch.float64, device=base_points.device
        )

        # Clone base_points and enable gradient computation
        base_points_copy = base_points.clone().detach().requires_grad_(True)
        for i in range(dim):
            for j in range(dim):
                base_points_copy.grad.zero_()
                g_ij = g[:, i, j]

                # Compute gradient of g_ij with respect to base_points
                grads = torch.autograd.grad(
                    outputs=g_ij,
                    inputs=base_points_copy,
                    grad_outputs=torch.ones_like(g_ij),
                    create_graph=True,
                )

                # Collect gradients for all points in the batch
                dg_dx[:, i, j, :] = grads[0]

        # Compute batch of Christoffel symbols
        christoffels = 0.5 * torch.einsum("bkl, bijl -> bijk", g_inv, dg_dx)

        return christoffels


class EuclideanMetric(RiemannianMetric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y, base_point=None):
        return torch.dot(x, y)


class PoincareMetric(RiemannianMetric):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(u, v, base_point):
        return (
            4 * torch.dot(u, v) / (1 - torch.linalg.vector_norm(base_point) ** 2) ** 2
        )


# TODO: make sure Sasaki metric is implemented correctly
#       in particular whether choice of splitting is canonical?
class SasakiMetric(RiemannianMetric):
    def __init__(self, base_metric: RiemannianMetric):
        super().__init__()
        self.base_metric = base_metric

    def forward(self, p, q, base_point=None):
        n = int(p.size(0) / 2)
        # split p into point on base manifold and tangent vector
        x = p[:n]
        v = p[n:]
        # split q into point on base manifold and tangent vector
        y = q[:n]
        u = q[n:]
        # split base_point into point on base manifold (and forget about tangent)
        if base_point is not None:
            if base_point.size(0) == 2 * n:
                base_point = base_point[:n]

        g_xy = self.base_metric.forward(x, y, base_point=base_point)
        g_uv = self.base_metric.forward(u, v, base_point=base_point)

        return g_xy + g_uv
