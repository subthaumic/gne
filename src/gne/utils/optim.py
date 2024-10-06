import torch


class RiemannianSGD(torch.optim.Optimizer):
    r"""Riemannian stochastic gradient descent optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        metric (RiemannianMetric): Function that returns the Riemannian
            metric tensor at a point
    """

    def __init__(self, params, **kwargs):
        super(RiemannianSGD, self).__init__(params, defaults=kwargs)

    def step(self, closure=None):
        """Performs optimization step using Euler approximation of geodesic curve.

        Args:
            closure (callable, optional): A closure that reevaluates the model and
                returns the loss.
        """
        metric = self.defaults["metric"]
        dt = self.defaults["lr"]

        loss = closure()

        for group in self.param_groups:
            if group["lr"]:
                dt = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # get points and gradients
                x = p.data
                print(len(x))
                christoffels = metric.get_christoffels(x)
                v = -p.grad.data
                if torch.any(torch.isnan(v)):
                    v[torch.isnan(v)] = 0

                v[torch.isnan(v)] = 0

                # Update parameter using leap-frog approximation of geodesic
                v -= torch.einsum("bijk,bi,bj->bk", christoffels, v, v) * dt / 2
                x += v * dt

        return loss


# TODO: If metric=Euclidean_metric, then use standard SGD
# Think about how things need to be set up to default to faster
# implementations in easy cases
