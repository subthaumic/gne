import numpy as np
import scipy
import sklearn

# import polars as pl


def connect_knn(KNN, distances, n_components, labels):
    """
    Given a KNN graph, connect nodes until we obtain a single connected
    component.
    """

    cur_comp = 0
    while n_components > 1:
        idx_cur = np.where(labels == cur_comp)[0]
        idx_rest = np.where(labels != cur_comp)[0]
        d = distances[idx_cur][:, idx_rest]
        ia, ja = np.where(d == np.min(d))
        i = ia
        j = ja

        KNN[idx_cur[i], idx_rest[j]] = distances[idx_cur[i], idx_rest[j]]
        KNN[idx_rest[j], idx_cur[i]] = distances[idx_rest[j], idx_cur[i]]

        nearest_comp = labels[idx_rest[j]]
        labels[labels == nearest_comp] = cur_comp
        n_components -= 1

    return KNN


def connected_neighbors_graph(
    features,
    mode="features",
    k_neighbours=15,
    distfn="sym",
    metric="minkowski",
):
    KNN = sklearn.neighbors.kneighbors_graph(
        features,
        k_neighbours,
        mode="distance",
        metric=metric,
        include_self=False,
    ).toarray()

    if "sym" in distfn.lower():
        KNN = np.maximum(KNN, KNN.T)
    else:
        KNN = np.minimum(KNN, KNN.T)

    n_components, labels = scipy.sparse.csgraph.connected_components(KNN)

    distances = sklearn.metrics.pairwise_distances(features, metric=metric)
    KNN = connect_knn(KNN, distances, n_components, labels)

    return KNN
