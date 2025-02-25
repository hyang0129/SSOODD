# Code adapted from
# https://github.com/facebookresearch/suncet

import torch

import os
import math
import torch
import torch.distributed as dist
from pdb import set_trace as pb


def init_paws_loss(
    multicrop=6,
    tau=0.1,
    T=0.25,
    me_max=True
):
    """
    Make semi-supervised PAWS loss

    :param multicrop: number of small multi-crop views
    :param tau: cosine similarity temperature
    :param T: target sharpenning temperature
    :param me_max: whether to perform me-max regularization
    """
    softmax = torch.nn.Softmax(dim=1)

    def sharpen(p):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
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

    def loss(
        anchor_views,
        anchor_supports,
        anchor_support_labels,
        target_views,
        target_supports,
        target_support_labels,
        sharpen=sharpen,
        snn=snn,
        **kwargs
    ):
        # -- NOTE: num views of each unlabeled instance = 2+multicrop
        batch_size = len(anchor_views) // (2+multicrop)

        # Step 1: compute anchor predictions
        probs = snn(anchor_views, anchor_supports, anchor_support_labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = snn(target_views, target_supports, target_support_labels)
            targets = sharpen(targets)
            if multicrop > 0:
                mc_target = 0.5*(targets[:batch_size]+targets[batch_size:])
                targets = torch.cat([targets, *[mc_target for _ in range(multicrop)]], dim=0)
            targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = torch.tensor(0.0).to(loss)
        if me_max:
            avg_probs = AllReduce.apply(torch.mean(sharpen(probs), dim=0))
            rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))

        return loss, rloss

    return loss


def make_labels_matrix(
    num_classes,
    s_batch_size,
    world_size,
    device,
    supervised_views=1,
    unique_classes=False,
    smoothing=0.0
):
    """
    Make one-hot labels matrix for labeled samples

    NOTE: Assumes labeled data is loaded with ClassStratifiedSampler from
          src/data_manager.py
    """

    local_images = s_batch_size*num_classes
    total_images = local_images*world_size

    off_value = smoothing/(num_classes*world_size) if unique_classes else smoothing/num_classes

    if unique_classes:
        labels = torch.zeros(total_images*supervised_views, num_classes*world_size).to(device) + off_value
        for r in range(world_size):
            for sv in range(supervised_views):
                # -- index range for rank 'r' images
                s1 = r * local_images * supervised_views + local_images * sv
                e1 = s1 + local_images
                # -- index offset for rank 'r' classes
                offset = r * num_classes 
                for i in range(num_classes):
                    labels[s1:e1][i::num_classes][:, offset+i] = 1. - smoothing + off_value
    else:
        labels = torch.zeros(local_images, num_classes).to(device) + off_value
        for i in range(num_classes):
            labels[i::num_classes][:, i] = 1. - smoothing + off_value
        labels = torch.cat([labels for _ in range(supervised_views)])
        labels = torch.cat([labels for _ in range(world_size)])

    # if dist.is_initialized() and dist.get_world_size() > 1:
    #     rank = dist.get_rank()
    # else:
    #     rank = 0
    # if rank == 0:
    #     pb()

    return labels

# ===============================================================

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads