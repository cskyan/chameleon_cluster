from graphtools import *
import itertools
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin


def internal_interconnectivity(graph, cluster):
    return np.sum(bisection_weights(graph, cluster))


def relative_interconnectivity(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    EC = np.sum(get_weights(graph, edges))
    ECci, ECcj = internal_interconnectivity(
        graph, cluster_i), internal_interconnectivity(graph, cluster_j)
    return EC / ((ECci + ECcj) / 2.0)


def internal_closeness(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = get_weights(cluster, edges)
    return np.sum(weights)


def relative_closeness(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        return 0.0
    else:
        SEC = np.mean(get_weights(graph, edges))
    Ci, Cj = internal_closeness(
        graph, cluster_i), internal_closeness(graph, cluster_j)
    SECci, SECcj = np.mean(bisection_weights(graph, cluster_i)), np.mean(
        bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


def merge_score(g, ci, cj, a):
    return relative_interconnectivity(
        g, ci, cj) * np.power(relative_closeness(g, ci, cj), a)


def merge_best(graph, df, a, k, verbose=False):
    clusters = np.unique(df['cluster'])
    max_score = 0
    ci, cj = -1, -1
    if len(clusters) <= k:
        return False

    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            if verbose:
                print("Checking c%d c%d" % (i, j))
            gi = get_cluster(graph, [i])
            gj = get_cluster(graph, [j])
            edges = connecting_edges(
                (gi, gj), graph)
            if not edges:
                continue
            ms = merge_score(graph, gi, gj, a)
            if verbose:
                print("Merge score: %f" % (ms))
            if ms > max_score:
                if verbose:
                    print("Better than: %f" % (max_score))
                max_score = ms
                ci, cj = i, j

    if max_score > 0:
        if verbose:
            print("Merging c%d and c%d" % (ci, cj))
        df.loc[df['cluster'] == cj, 'cluster'] = ci
        for i, p in enumerate(graph.nodes()):
            if graph.nodes[p]['cluster'] == cj:
                graph.nodes[p]['cluster'] = ci
    return max_score > 0


def cluster(df, n_clusters, knn_params=dict(n_neighbors=10, metric='euclidean', n_jobs=1), m=30, alpha=2.0, verbose=False, plot=False):
    graph = knn_graph(df, knn_params, verbose=True)
    graph = pre_part_graph(graph, m, df, verbose=True)
    iterm = tqdm(enumerate(range(m - n_clusters)), total=m-n_clusters)
    for i in iterm:
        merge_best(graph, df, alpha, n_clusters, verbose)
        if plot:
            plot2d_data(df)
    res = rebuild_labels(df)
    return res

def rebuild_labels(df, offset=0):
    ans = df.copy()
    clusters = list(pd.DataFrame(df['cluster'].value_counts()).index)
    c = offset
    for i in clusters:
        ans.loc[df['cluster'] == i, 'cluster'] = c
        c = c + 1
    return ans


class Chameleon(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=8, m=30, alpha=2.0, knn_neighbors=20, metric='minkowski', p=2, metric_params=None, n_jobs=None, verbose=False):
        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.knn_params = dict(n_neighbors=knn_neighbors, mode='connectivity', metric=metric, p=p, metric_params=metric_params, include_self='auto', n_jobs=n_jobs)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        res = cluster(pd.DataFrame(X), n_clusters=self.n_clusters, knn_params=self.knn_params, m=self.m, alpha=self.alpha, verbose=self.verbose, plot=False)
        self.labels_ = res['cluster'].values
        return self

    def fit_predict(self, X, y=None):
        return super().fit_predict(X, y)
