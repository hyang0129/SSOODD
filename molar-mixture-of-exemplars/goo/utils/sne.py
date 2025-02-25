
from pdb import set_trace as pb
from typing import Any
import torch
import pandas as pd
import re
import numpy as np
import math
import copy
import faiss


# https://github.com/mxl1990/tsne-pytorch/blob/master/tsne_torch.py

def SNE_P_dist(query, support, tol=1e-5, perplexity=30.0, starting_kappa = 10.0, max_tries = 50, mask=None, kernel='vmf'):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    if kernel == 'vmf':
        query = torch.nn.functional.normalize(query)
        support = torch.nn.functional.normalize(support)
        
        D = query @ support.T
    elif kernel == 'gaussian':
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
        D = -(torch.sum(query**2, dim=1)[:, None] + torch.sum(support**2, dim=1)[None,:] - 2 * (query @ support.T))

    LARGE_NUM = 1e8
    if mask is not None:
        D = D - mask*LARGE_NUM
    else:
        if query.shape == support.shape:
            mask = torch.nn.functional.one_hot(torch.arange(query.shape[0]), query.shape[0]).to(query)
            D = D - mask*LARGE_NUM

    P = torch.zeros_like(D)
    n = D.shape[0]
    kappa = torch.ones(n, 1).to(D) * starting_kappa
    new_kappa = None
    logU = torch.log(torch.tensor([perplexity])).to(query)
    n_list = [i for i in range(n)]
    softmax = torch.nn.Softmax(dim=1)

    Hdiff_max = 1
    eps = 1e-20

    tries = 0
    kappamin = kappa.clone()
    kappamin[:,:] = 0
    kappamax = kappa.clone()
    kappamax[:,:] = float('inf')

    while Hdiff_max > tol and tries < max_tries:
        if new_kappa is not None:
            kappa = new_kappa

        P = softmax(D*kappa)
        H = - torch.sum(P*torch.log(P+eps), dim=1, keepdim=True)
        
        Hdiff = H - logU
        Hdiff_max = torch.abs(Hdiff).max()
        kappamin[Hdiff>0] = kappa[Hdiff>0]
        kappamax[Hdiff<=0] = kappa[Hdiff<=0]
        # =====================================
        kappa_max1 = kappa*2.0
        kappa_max2 = (kappa + kappamax)/2.0

        kappa_min1 = kappa/2
        kappa_min2 = (kappa + kappamin) / 2.0

        kappa_min_use = torch.max(kappa_min1, kappa_min2)
        kappa_max_use = torch.min(kappa_max1, kappa_max2)

        new_kappa = kappa.clone()
        new_kappa[Hdiff<=0] = kappa_min_use[Hdiff<=0]
        new_kappa[Hdiff>0] = kappa_max_use[Hdiff>0]
        tries += 1

    # Return final P-matrix
    P = softmax(D*new_kappa)
    return P, new_kappa, Hdiff_max, tries

def SNE_Q_dist(query, support, temp, mask=None, kernel='vmf'):

    if kernel == 'vmf':
        query = torch.nn.functional.normalize(query)
        support = torch.nn.functional.normalize(support)
        
        D = query @ support.T
    elif kernel == 'gaussian':
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
        D = -(torch.sum(query**2, dim=1)[:, None] + torch.sum(support**2, dim=1)[None,:] - 2 * (query @ support.T))

    softmax = torch.nn.Softmax(dim=1)

    LARGE_NUM = 1e8
    if mask is not None:
        D = D - mask*LARGE_NUM
    else:
        if query.shape == support.shape:
            mask = torch.nn.functional.one_hot(torch.arange(query.shape[0]), query.shape[0]).to(query)
            D = D - mask*LARGE_NUM

    p12 = softmax(D/temp)
    return p12

# ======================================
# ====================================================================

