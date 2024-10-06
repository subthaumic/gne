import timeit
import torch
import scipy
import numpy as np


class AffinityTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()


class RandomForestAffinity(AffinityTransform):
    def init(
        graph,
        sigma=1.0,
        metric="minkowski",
    ):
        """
        Computes the target RFA similarity matrix. The RFA matrix of
        similarities relates to the commute time between pairs of nodes, and it is
        built on top of the Laplacian of a single connected component k-nearest
        neighbour graph of the data.
        """
        start = timeit.default_timer()

        # if metric == "minkowski" and \
        #  isinstance(features, pl.dataframe.frame.DataFrame):
        #     S = np.exp(-KNN / (sigma * features.width * features.height))
        # elif metric == "minkowski":
        #     S = np.exp(-KNN / (sigma * features.size(1)))
        # else:
        S = np.exp(-graph / sigma)

        S[graph == 0] = 0
        print("Computing laplacian...")
        L = scipy.sparse.csgraph.laplacian(S, normed=False)
        print(f"Laplacian computed in {(timeit.default_timer() - start):.2f} sec")

        print("Computing RFA...")
        start = timeit.default_timer()
        RFA = np.linalg.inv(L + np.eye(L.shape[0]))
        RFA[RFA == np.nan] = 0.0

        print(f"RFA computed in {(timeit.default_timer() - start):.2f} sec")

        return torch.Tensor(RFA)
