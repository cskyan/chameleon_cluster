import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
from visualization import *

import metis


def knn_graph(df, knn_params=dict(n_neighbors=2, metric="euclidean"), verbose=False):
    if knn_params.setdefault('metric', "euclidean") == 'precomputed':
        g = nx.from_scipy_sparse_matrix(knn_params['distance'])
    else:
        points = [p[1:] for p in df.itertuples()]
        knn_params["mode"] = "distance"
        knn_params["n_neighbors"] = min(knn_params["n_neighbors"], df.shape[0]-1)
        if verbose:
            print("Building kNN graph (k = %d)..." % (knn_params.setdefault("n_neighbors", 2)))
        A = kneighbors_graph(points, **knn_params)
        g = nx.from_scipy_sparse_matrix(A)
        nx.set_node_attributes(g, {n: {"pos": points[i]} for i, n in enumerate(g.nodes())})
    g.graph["edge_weight_attr"] = "similarity"
    return g


def part_graph(graph, k, df=None):
    edgecuts, parts = metis.part_graph(
        graph, 2, objtype="cut", ufactor=250)
    # print(edgecuts)
    for i, p in enumerate(graph.nodes()):
        graph.nodes[p]["cluster"] = parts[i]
    if df is not None:
        df["cluster"] = nx.get_node_attributes(graph, "cluster").values()
    return graph


def pre_part_graph(graph, k, df=None, verbose=False):
    if verbose:
        print("Begin clustering...")
    clusters = 0
    for i, p in enumerate(graph.nodes()):
        graph.nodes[p]["cluster"] = 0
    cnts = {}
    cnts[0] = len(graph.nodes())

    while clusters < k - 1:
        maxc = -1
        maxcnt = 0
        for key, val in cnts.items():
            if val > maxcnt:
                maxcnt = val
                maxc = key
        s_nodes = [n for n in graph.nodes if graph.nodes[n]["cluster"] == maxc]
        s_graph = graph.subgraph(s_nodes)
        edgecuts, parts = metis.part_graph(
            s_graph, 2, objtype="cut", ufactor=250)
        new_part_cnt = 0
        for i, p in enumerate(s_graph.nodes()):
            if parts[i] == 1:
                graph.nodes[p]["cluster"] = clusters + 1
                new_part_cnt = new_part_cnt + 1
        cnts[maxc] = cnts[maxc] - new_part_cnt
        cnts[clusters + 1] = new_part_cnt
        clusters = clusters + 1

    edgecuts, parts = metis.part_graph(graph, k)
    if df is not None:
        df["cluster"] = nx.get_node_attributes(graph, "cluster").values()
    return graph


def get_cluster(graph, clusters):
    nodes = [n for n in graph.nodes if graph.nodes[n]["cluster"] in clusters]
    return nodes


def connecting_edges(partitions, graph):
    cut_set = []
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph:
                if b in graph[a]:
                    cut_set.append((a, b))
    return cut_set


def min_cut_bisector(graph):
    graph = graph.copy()
    graph = part_graph(graph, 2)
    partitions = get_cluster(graph, [0]), get_cluster(graph, [1])
    return connecting_edges(partitions, graph)


def get_weights(graph, edges):
    return [graph[edge[0]][edge[1]]["weight"] for edge in edges]


def bisection_weights(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = min_cut_bisector(cluster)
    weights = get_weights(cluster, edges)
    return weights
