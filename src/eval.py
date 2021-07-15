# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import numpy as np
from typing import Sequence, Tuple, List
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

THRE = 0.5


def fair_link_eval(
        emb: np.ndarray,
        sensitive: np.ndarray,
        test_edges_true: Sequence[Tuple[int, int]],
        test_edges_false: Sequence[Tuple[int, int]],
        rec_ratio: List[float] = None,
) -> Sequence[List]:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.array(np.dot(emb, emb.T), dtype=np.float128)

    preds_pos_intra = []
    preds_pos_inter = []
    for e in test_edges_true:
        if sensitive[e[0]] == sensitive[e[1]]:
            preds_pos_intra.append(sigmoid(adj_rec[e[0], e[1]]))
        else:
            preds_pos_inter.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg_intra = []
    preds_neg_inter = []
    for e in test_edges_false:
        if sensitive[e[0]] == sensitive[e[1]]:
            preds_neg_intra.append(sigmoid(adj_rec[e[0], e[1]]))
        else:
            preds_neg_inter.append(sigmoid(adj_rec[e[0], e[1]]))

    res = {}
    for preds_pos, preds_neg, type in zip((preds_pos_intra, preds_pos_inter, preds_pos_intra + preds_pos_inter),
                                          (preds_neg_intra, preds_neg_inter, preds_neg_intra + preds_neg_inter),
                                          ("intra", "inter", "overall")):
        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        err = (np.sum(list(map(lambda x: x >= THRE, preds_pos))) + np.sum(
            list(map(lambda x: x < THRE, preds_neg)))) / (len(preds_pos) + len(preds_neg))

        score_avg = (sum(preds_pos) + sum(preds_neg)) / (len(preds_pos) + len(preds_neg))
        pos_avg, neg_avg = sum(preds_pos) / len(preds_pos), sum(preds_neg) / len(preds_neg)

        res[type] = [roc_score, ap_score, err, score_avg, pos_avg, neg_avg]

    ks_pos = stats.ks_2samp(preds_pos_intra, preds_pos_inter)[0]
    ks_neg = stats.ks_2samp(preds_neg_intra, preds_neg_inter)[0]

    standard = res["overall"][0:2] + [abs(res["intra"][i] - res["inter"][i]) for i in range(3, 6)] + [ks_pos, ks_neg]

    return standard
