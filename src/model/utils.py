# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Tuple


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape

    return coords, values, shape


def preprocess_graph(adj):
    """ D^(-1) * A """

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))

    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()

    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor """

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def get_score(emb: np.ndarray, test_edges_true: List, test_edges_false: List) -> Tuple[float, float]:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)

    preds_pos = []
    for e in test_edges_true:
        preds_pos.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in test_edges_false:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def project(y: torch.tensor):
    des, _ = torch.sort(y, descending=True)
    cumsum = torch.cumsum(des, dim=0)
    pos = (torch.ones(des.shape[0]) / torch.arange(1, des.shape[0] + 1)).cuda()

    rho = des + pos * (torch.ones(des.shape[0]).to(y.device) - cumsum)
    rho = (rho > 0).sum()
    lambda_ = (1. / rho.float()) * (1. - cumsum[rho - 1])

    x = y + lambda_
    x[x < 0.] = 0.

    return x
