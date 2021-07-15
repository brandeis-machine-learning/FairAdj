# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import numpy as np
from collections import Counter
import networkx as nx
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

cora_label = {
    "Genetic_Algorithms": 0,
    "Reinforcement_Learning": 1,
    "Neural_Networks": 2,
    "Rule_Learning": 3,
    "Case_Based": 4,
    "Theory": 5,
    "Probabilistic_Methods": 6,
}


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value][0]


def cora(feat_path="../data/cora/cora.content", edge_path="../data/cora/cora.cites", scale=True,
         test_ratio=0.1) -> Tuple:
    idx_features_labels = np.genfromtxt(feat_path, dtype=np.dtype(str))
    idx_features_labels = idx_features_labels[idx_features_labels[:, 0].astype(np.int32).argsort()]

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    nodelist = {idx: node for idx, node in enumerate(idx)}
    X = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    sensitive = np.array(list(map(cora_label.get, idx_features_labels[:, -1])))

    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    G = nx.read_edgelist(edge_path, nodetype=int)
    G, test_edges_true, test_edges_false = build_test(G, nodelist, test_ratio)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    adj = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))

    return G, adj, X, sensitive, test_edges_true, test_edges_false, nodelist


def build_test(G: nx.Graph, nodelist: Dict, ratio: float) -> Tuple:
    """
    Split training and testing set for link prediction in graph G.
    :param G: nx.Graph
    :param nodelist: idx -> node_id in nx.Graph
    :param ratio: ratio of positive links that used for testing
    :return: Graph that remove all test edges, list of index for test edges
    """

    edges = list(G.edges.data(default=False))
    num_nodes, num_edges = G.number_of_nodes(), G.number_of_edges()
    num_test = int(np.floor(num_edges * ratio))
    test_edges_true = []
    test_edges_false = []

    # generate false links for testing
    while len(test_edges_false) < num_test:
        idx_u = np.random.randint(0, num_nodes - 1)
        idx_v = np.random.randint(idx_u, num_nodes)

        if idx_u == idx_v:
            continue
        if (nodelist[idx_u], nodelist[idx_v]) in G.edges(nodelist[idx_u]):
            continue
        if (idx_u, idx_v) in test_edges_false:
            continue
        else:
            test_edges_false.append((idx_u, idx_v))

    # generate true links for testing
    all_edges_idx = list(range(num_edges))
    np.random.shuffle(all_edges_idx)
    test_edges_true_idx = all_edges_idx[:num_test]
    for test_idx in test_edges_true_idx:
        u, v, _ = edges[test_idx]
        G.remove_edge(u, v)
        test_edges_true.append((get_key(nodelist, u), get_key(nodelist, v)))

    return G, test_edges_true, test_edges_false


def analyze(G: nx.Graph, sensitive: np.ndarray) -> None:
    """ Show the base link rate on different sensitive attributes """

    mapping = {node: att for node, att in zip(sorted(G.nodes), sensitive)}
    count_node = Counter(sensitive)
    group_size = [size for size in count_node.values()]
    edge_type = []

    for edge in G.edges:
        if mapping[edge[0]] == mapping[edge[1]]:
            edge_type.append("intra")
        else:
            edge_type.append("inter")
    count_edge = Counter(edge_type)

    intra_base = sum([s * (s - 1) / 2 for s in group_size])
    inter_base = sum([s * (sum(group_size) - s) / 2 for s in group_size])

    print("node: {}".format(count_node))
    print("total nodes: {}".format(sum(count_node.values())))
    print("edge: {}".format(count_edge))
    print("total edges: {}".format(sum(count_edge.values())))
    print("base rate")
    print("intra: {:.5f}".format(float(count_edge["intra"]) / intra_base))
    print("inter: {:.5f}".format(float(count_edge["inter"]) / inter_base))

    return


if __name__ == "__main__":
    G, adj, X, sensitive, test_edges_true, test_edges_false, nodelist = cora()
    analyze(G, sensitive)
