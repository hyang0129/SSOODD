# Code adapted from
# https://github.com/facebookresearch/suncet

from logging import getLogger
import math
import torch

from .paws import AllGather, AllReduce
import torch.distributed as dist

from pdb import set_trace as pb
import copy

import numpy as np

logger = getLogger()


def init_molar_loss(
    # paws configs
    multicrop=6,
    tau=0.1,
    T=0.25,
    me_max=True,
    # ropaws configs
    ropaws=False,
    prior_tau=3.0,
    prior_pow=1.0,
    label_ratio=5.0,
    s_batch_size=6720,
    u_batch_size=4096,
    sharpen_func = 'consistency',
    prototype_contrast = False,
    prototype_tau = 0.5
):
    """
    Make semi-supervised PAWS loss

    :param multicrop: number of small multi-crop views
    :param tau: cosine similarity temperature
    :param T: target sharpenning temperature
    :param me_max: whether to perform me-max regularization
    """
    softmax = torch.nn.Softmax(dim=1)


    def sharpen_orig(p, **kwargs):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    # sharpen the mean # mixmatch
    def sharpen_mixmatch(p, unlabelled_global_views=2, batch_size=128):
        p = p.reshape(unlabelled_global_views, batch_size, -1)
        exponent = T
        sharp_p = torch.mean(p, dim=0)
        sharp_p = sharp_p**(1./exponent)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        sharp_p = sharp_p.repeat(unlabelled_global_views,1)
        return sharp_p

    def snn(query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 2: gather embeddings from all workers
        supports = AllGather.apply(supports)

        # Step 3: compute similarlity between local embeddings
        return softmax(query @ supports.T / tau) @ labels

    def snn_semi(query, supports, labels, n_views):
        """ Semi-supervised density estmation """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 2: gather embeddings from all workers
        supports = AllGather.apply(supports)

        # Step 3: compute similarlity between local embeddings
        probs = []
        for q in query.chunk(n_views):  # for each view
            p = _snn_semi_each(q, supports, labels)
            probs.append(p)
        probs = torch.cat(probs, dim=0)  # concat over views

        # Step 4: convert p_out probability to uniform
        M, C = probs.shape
        p_in = probs.sum(dim=1)
        unif = torch.ones(M, C, device=probs.device) / C
        probs = probs + (1 - p_in.view(M, 1)) * unif

        p_in = sum(p_in.chunk(n_views)) / n_views  # average over views

        return probs, p_in

    def _snn_semi_each(query, supports, labels):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        query = AllGather.apply(query)
        M = query.size(0)
        N, K = labels.size()

        device = query.device
        arange = lambda n: torch.arange(n, device=device)
        eye = lambda n: torch.eye(n, n, device=device)

        # compute similarity matrix
        s_sims = query @ supports.T  # cosine sims MxN
        u_sims = query @ query.T  # cosine sims MxM
        u_sims[arange(M), arange(M)] = -float('inf')  # remove self-sims

        # compute in-domain prior
        max_sim = s_sims.max(dim=1)[0]
        prior = ((max_sim - 1) / prior_tau).exp().view(M, 1)

        # compute pseudo-label
        r = label_ratio * u_batch_size / s_batch_size
        s_sims = s_sims + math.log(r) * tau  # upscale labeled batch

        C = softmax(torch.cat([s_sims, u_sims], dim=1) / tau) * prior  # Mx(N+M)
        C_L, C_U = C[:, :N], C[:, N:]  # MxN, MxM
        probs = torch.linalg.inv(eye(M) - C_U) @ C_L @ labels  # MxK

        # get values for this node
        probs = probs[ M//world_size * rank : M//world_size * (rank + 1) ]
        return probs

    def snn_same(query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 2: gather embeddings from all workers
        supports = AllGather.apply(supports)

        # Step 3: compute similarlity between local embeddings
        logits = query @ supports.T / prototype_tau
        logits.fill_diagonal_(-1e4)
        return softmax(logits) @ labels


    if sharpen_func == 'consistency':
        sharpen_use = sharpen_orig
    elif sharpen_func == 'mixmatch':
        sharpen_use = sharpen_mixmatch

    def loss(
        anchor_views,
        anchor_view_labels,
        anchor_supports,
        anchor_support_labels,
        target_views,
        target_supports,
        target_support_labels,
        target_support_labels_index,
        supervised=False,
        sharpen=sharpen_use,
        snn=snn,
        snn_semi=snn_semi,
        prototype_contrast=prototype_contrast,
        **kwargs
    ):
        # -- NOTE: num views of each unlabeled instance = 2+multicrop
        batch_size = len(anchor_views) // (2+multicrop)

        # Step 1: compute anchor predictions
        probs = snn(anchor_views, anchor_supports, anchor_support_labels)

        # Step 2: compute targets for anchor predictions

        if supervised:
            # if dist.get_rank() == 0:
            #     pb()
            # pb()
            target_support_labels_index = AllGather.apply(target_support_labels_index)

            mask = torch.isin(anchor_view_labels, target_support_labels_index.unique())
            probs = probs[mask.repeat((2+multicrop))]
            anchor_view_labels = anchor_view_labels[mask]

            # need to figure out index order of support targets, and map the anchor_view_labels to this order
            unique_elements, inverse_indices = np.unique(target_support_labels_index.cpu(), return_index=True)
            
            # support_labels = target_support_labels_index[:target_support_labels.shape[1]]
            support_labels = torch.tensor(unique_elements[inverse_indices.argsort()]).to(target_support_labels_index)

            index_order = support_labels.argsort()
            mapping = torch.zeros(support_labels.max()+1).long().to(support_labels)
            mapping[support_labels[index_order]] = index_order

            targets = torch.nn.functional.one_hot(mapping[anchor_view_labels], num_classes=target_support_labels.shape[1]).repeat(2+multicrop, 1)
        else:
            with torch.no_grad():
                if ropaws:
                    targets, p_in = snn_semi(target_views, target_supports, target_support_labels, n_views=2)
                else:
                    targets = snn(target_views, target_supports, target_support_labels)
                targets = sharpen(targets, unlabelled_global_views=2, batch_size=batch_size)
                if multicrop > 0:
                    mc_target = 0.5*(targets[:batch_size]+targets[batch_size:])
                    targets = torch.cat([targets, *[mc_target for _ in range(multicrop)]], dim=0)
                targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        if ropaws:
            weight = p_in.repeat(2 + multicrop) ** prior_pow  # weighted loss
            loss = torch.mean(weight * torch.sum(-targets * torch.log(probs), dim=1))
        else:
            loss = torch.mean(torch.sum(torch.log(probs ** (-targets)), dim=1))

        if prototype_contrast:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size = 1
                rank = 0
            prototype_batch_size = anchor_supports.shape[0]
            prototype_targets = (anchor_support_labels > 0.5).float()
            prototype_probs = snn_same(anchor_supports, anchor_supports, prototype_targets)
            prototype_targets = prototype_targets[rank*prototype_batch_size:(rank+1)*prototype_batch_size]
            loss += torch.mean(torch.sum(-prototype_targets*torch.log(prototype_probs), dim=1))

        # Step 4: compute me-max regularizer
        rloss = torch.tensor(0.0).to(loss)
        if me_max:
            sharp_probs = sharpen(probs, unlabelled_global_views=(2+multicrop), batch_size=batch_size)
            avg_probs = AllReduce.apply(torch.mean(sharp_probs, dim=0))
            rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))
        return loss, rloss

    return loss

