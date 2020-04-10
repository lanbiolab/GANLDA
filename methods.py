from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


# loss function
def loss_function(pre_adj, adj):
    adj = torch.Tensor(adj)
    loss_fn = torch.nn.BCELoss()
    return loss_fn(pre_adj, adj)


# train method
def train(label, ganlda_model, optimizer, lncx, disx, adj):

    # train
    optimizer.zero_grad()
    pred = ganlda_model(lncx, disx, adj)
    loss = loss_function(pred, label)
    loss.backward()
    optimizer.step()

    return pred, loss


# sort the score matrix
def sort_matrix(score_matrix, interact_matrix):
    sort_index = np.argsort(-score_matrix, axis=0)
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted