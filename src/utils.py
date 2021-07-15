# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, List
import torch


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return


def save(path: str, columns: List[str], data: List[str or float]) -> None:
    if not os.path.isfile(path):
        res = pd.DataFrame(columns=columns)
    else:
        res = pd.read_csv(path)

    curr_res = pd.DataFrame([data], columns=columns)
    res = pd.concat([res, curr_res])
    res.to_csv(path, index=False)

    return


def find_link(adj: sp.coo_matrix, sensitive: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Locate the intra and inter links (or positive links) in the adjacency matrix """

    binary = True if max(sensitive) == 1 else False
    link_pos = np.array([adj.row, adj.col]).transpose()

    if binary:
        sensitive = sensitive * 2 - 1  # turn sensitive from [0, 1] to [-1, +1]
        sensitive = sensitive[:, np.newaxis]
        sensitive_mat = np.dot(sensitive, sensitive.transpose())

        intra_pos = np.asarray(np.where(sensitive_mat == 1)).transpose()
        inter_pos = np.asarray(np.where(sensitive_mat == -1)).transpose()

        link_type = sensitive_mat[link_pos[:, 0], link_pos[:, 1]]
        intra_link_pos = link_pos[np.where(link_type == 1)[0], :]
        inter_link_pos = link_pos[np.where(link_type == -1)[0], :]
    else:
        intra_link_pos = inter_link_pos = intra_pos = inter_pos = np.empty((0, 2)).astype(np.int32)
        for s in set(sensitive):
            sensitive_copy = sensitive.copy()
            sensitive_copy[sensitive_copy != s] = -1
            sensitive_copy[sensitive_copy == s] = -2
            sensitive_copy = sensitive_copy[:, np.newaxis]
            sensitive_mat = np.dot(sensitive_copy, sensitive_copy.transpose())

            intra_pos_s = np.asarray(np.where(sensitive_mat == 4)).transpose()
            inter_pos_s = np.asarray(np.where(sensitive_mat == 2)).transpose()

            intra_pos = np.concatenate([intra_pos, intra_pos_s], axis=0)
            inter_pos = np.concatenate([inter_pos, inter_pos_s], axis=0)

            link_type = sensitive_mat[link_pos[:, 0], link_pos[:, 1]]
            intra_link_pos_s = link_pos[np.where(link_type == 4)[0], :]
            inter_link_pos_s = link_pos[np.where(link_type == 2)[0], :]

            intra_link_pos = np.concatenate([intra_link_pos, intra_link_pos_s], axis=0)
            inter_link_pos = np.concatenate([inter_link_pos, inter_link_pos_s], axis=0)

    return intra_pos, inter_pos, intra_link_pos, inter_link_pos
