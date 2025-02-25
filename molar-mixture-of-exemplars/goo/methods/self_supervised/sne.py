
from ..model_base_s import ModelBaseS
from pdb import set_trace as pb
from typing import Any
import torch
import pandas as pd
import re
import numpy as np
import math
import copy
import torch.distributed as dist
from ...utils.sne import SNE_P_dist, SNE_Q_dist

from ...loss.paws import AllGather, AllReduce

class SNE(ModelBaseS):
    def __init__(self,
            perplexity = 30,
            target_temperature = 0.1,
            kernel = 'vmf',
            **kwargs):
        super(SNE, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.metrics = ['index', 'label', 'tsne_loss', 'Hdiff_max', 'max_tries', 'kappa']
        self.metrics_log = [False, False,    True,    True,    True,    True]
        self.metrics_save = [True, True,     False,     False,     False,     False]

        self.backbone = self.networks.backbone()
        self.head_semi = self.networks.head()
        self.backbone_head_semi = torch.nn.Sequential(
                self.backbone,
                self.head_semi
            )
        self.mean_kappa = 1

    def evaluate_image_list(self, inputs, return_before_head=False, prototypes=False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _z, _h = self.forward(torch.cat(inputs[start_idx:end_idx]))
            if start_idx == 0:
                h, z = _h, _z
            else:
                h, z = torch.cat((h, _h)), torch.cat((z, _z))
            start_idx = end_idx

        if return_before_head:
            return h, z
        return h

    def model_step(self, batch, stage='fit'):

        imgs, y, idx = batch
        h, zss = self.evaluate_image_list(imgs, return_before_head=True)

        with torch.cuda.amp.autocast(enabled=False):

            if self.trainer.precision == '64-true':
                h, zss = h, zss
            else:
                h, zss = h.float(), zss.float()

            anchor_views_n = h

            anchor_views_n_large = AllGather.apply(anchor_views_n)
            zss_large = AllGather.apply(zss)

            batch_size = zss.shape[0]
            enlarged_batch_size = zss_large.shape[0]
            if enlarged_batch_size != batch_size:
                rank = dist.get_rank()
                labels_oh = torch.arange(batch_size, device=zss.device) + rank * batch_size
                masks = torch.nn.functional.one_hot(labels_oh, enlarged_batch_size).to(zss)
            else:
                masks = None

            zss_all = AllGather.apply(zss)
            anchor_views_n_all = AllGather.apply(anchor_views_n)

            p12 = SNE_Q_dist(anchor_views_n, anchor_views_n_all, self.hparams.target_temperature, mask=masks, kernel=self.hparams.kernel)
            pbb12, kappa, Hdiff_max, tries = SNE_P_dist(zss, zss_all, tol=1e-5, perplexity=self.hparams.perplexity, 
                    starting_kappa = self.mean_kappa, mask=masks, kernel=self.hparams.kernel)
            self.mean_kappa = kappa.mean()

            p12_all = AllGather.apply(p12)
            pbb12_all = AllGather.apply(pbb12)
            P = pbb12_all + pbb12_all.T
            P = P/P.sum()
            Q = p12_all + p12_all.T
            Q = Q/Q.sum()

            eps = 1e-20
            Q = torch.maximum(Q, torch.tensor(eps))
            tsne_loss = torch.sum(P*torch.log(P/Q+eps))


        results_dict = {
            'loss': tsne_loss, 'index': idx, 'label': y, 'tsne_loss': tsne_loss,
            'Hdiff_max': Hdiff_max.mean(), 'max_tries': torch.tensor(tries).float(), 'kappa': kappa.mean()
        }
        return results_dict

    def forward(self, x):
        embed = self.backbone(x).flatten(start_dim=1)
        z_semi = self.head_semi(embed)
        return embed, z_semi

    def predict_step(self, batch: Any, batch_idx: int):
        x = batch[0][0]
        idx = batch[2]
        res = self.forward(x)
        return res, idx
