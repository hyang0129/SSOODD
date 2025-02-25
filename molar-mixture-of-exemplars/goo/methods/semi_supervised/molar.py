
from ..model_base_s import ModelBaseS
from pdb import set_trace as pb
from typing import Any
import torch
import pandas as pd
import re
import numpy as np
import math
import copy
from ...callbacks.paws_accuracy import PAWSPrediction
from ...loss.paws import make_labels_matrix
import torch.distributed as dist
import os
from pathlib import Path

class MoLAR(ModelBaseS):
    def __init__(self, label_smoothing = 0.1, 
            supervised_views = 2,
            supervised=True,
            save_bank=False,
            **kwargs):
        super(MoLAR, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.metrics = ['index', 'label', 'y_hat', 'loss_semi', 'ploss', 'me_max']
        self.metrics_log = [False, False,    False,    True,     True,    True]
        self.metrics_save = [True, True,     True,     False,    False,   False]

        self.backbone = self.networks.backbone()
        self.head_semi = self.networks.head()
        self.backbone_head_semi = torch.nn.Sequential(
                self.backbone,
                self.head_semi
            )

        self.paws_prediction = PAWSPrediction(backbone='backbone_head_semi')

    def setup(self, stage):
        super(MoLAR, self).setup(stage)
        trainer_state = copy.deepcopy(self.trainer.state)
        self.labelled_indices = None
        self.paws_prediction.setup(self.trainer, self, stage)
        init = self.trainer.estimated_stepping_batches
        sampler = self.trainer.train_dataloader['labelled'].batch_sampler

        self.unique_classes= sampler.unique_cpb
        self.classes_per_batch= sampler.cpb
        self.effective_classes = sampler.effective_classes

        if torch.distributed.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1
        self.s_batch_size = sampler.batch_size #self.classes
        self.u_batch_size = max(int(self.trainer.datamodule.hparams.batch_size*self.trainer.datamodule.hparams.batch_size_unlabelled_scalar//self.world_size), 1)

        self.labels_matrix = make_labels_matrix(
                num_classes= self.classes_per_batch, # self.classes,
                s_batch_size=self.s_batch_size,
                supervised_views = self.hparams.supervised_views,
                world_size=self.world_size,
                device=self.device,
                unique_classes=self.unique_classes,
                smoothing=self.hparams.label_smoothing
            )
        tmp = math.ceil(len(self.trainer.train_dataloader['unlabelled']) / len(sampler))
        self.trainer.datamodule.hparams.train_labelled_batch_sampler.keywords['epochs'] = tmp

        self.trainer.fit_loop._combined_loader = None
        self.trainer.fit_loop.setup_data()
        self.trainer.state = trainer_state

    def evaluate_image_list(self, inputs, return_before_head=False):
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
        if stage != 'fit':
            [x], y, idx = batch
            y_hat = self.backbone_head_semi(x)
            y_hat_prob = self.paws_prediction.evaluate_batch(y_hat, self.train_feature_bank_all, self.train_targets_bank_all)
            ploss = torch.tensor([0.0])
            me_max = torch.tensor([0.0])
            loss_semi = torch.tensor([0.0])
        else:
            batch_labelled, y, idx = batch['labelled']
            batch_unlabelled, y_unlabelled, _ = batch['unlabelled']

            self.labels_matrix = self.labels_matrix.to(batch_labelled[0])
            # labels = torch.cat([self.labels_matrix for _ in range(self.hparams.supervised_views)])
            labels = self.labels_matrix
            imgs = batch_labelled + batch_unlabelled
            y_labelled = y

            h, zss = self.evaluate_image_list(imgs, return_before_head=True)

            # Compute paws loss in full precision
            with torch.cuda.amp.autocast(enabled=False):

                # Step 1. convert representations to fp32
                if self.trainer.precision == '64-true':
                    h, zss = h, zss
                else:
                    h, zss = h.float(), zss.float()

                # Step 2. determine anchor views/supports and their
                #         corresponding target views/supports
                # --
                num_support = self.hparams.supervised_views * self.s_batch_size * self.classes_per_batch
                # --
                anchor_supports = h[:num_support]
                anchor_views = h[num_support:]
                # --
                target_supports = h[:num_support].detach()
                target_views = h[num_support:].detach()
                target_views = torch.cat([
                    target_views[self.u_batch_size:2*self.u_batch_size],
                    target_views[:self.u_batch_size]], dim=0)

                # Step 3. compute paws loss with me-max regularization
                (ploss, me_max) = self.loss(
                    anchor_views=anchor_views,
                    anchor_view_labels=y_unlabelled,
                    anchor_supports=anchor_supports,
                    anchor_support_labels=labels,
                    target_views=target_views,
                    target_supports=target_supports,
                    target_support_labels=labels,
                    target_support_labels_index=y_labelled,
                    supervised=self.hparams.supervised,
                    idx=idx)
                # =========================================
                loss_semi = ploss + me_max
                y_hat_prob = torch.tensor([0.0])

        results_dict = {
            'loss': loss_semi, 'index': idx, 'label': y, 'ploss': ploss, 'me_max': me_max,
            'loss_semi': loss_semi, 'y_hat': y_hat_prob
        }
        return results_dict

    # ==========================================================
    def on_validation_epoch_start(self):
        train_data, test_data = self.paws_prediction.setup_data(self.trainer, None, basedata='train_labelled')
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        self.train_feature_bank_all, self.train_targets_bank_all = self.paws_prediction._evaluate_embeddings(train_data, self.trainer, rank)
        self.train_targets_bank_all = torch.nn.functional.one_hot(self.train_targets_bank_all, num_classes=self.num_classes).to(self.train_feature_bank_all)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        if self.hparams.save_bank:
            logger = self.trainer.logger
            output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'predictions')
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            self.train_feature_bank_all.cpu().numpy().dump(output_folder + '/train_feature_bank_all.npy')
            self.train_targets_bank_all.cpu().numpy().dump(output_folder + '/train_targets_bank_all.npy')

    # ==========================================================
    def forward(self, x):
        embed = self.backbone(x).flatten(start_dim=1)
        z_semi = self.head_semi(embed)
        return embed, z_semi

    def predict_step(self, batch: Any, batch_idx: int):
        x = batch[0][0]
        idx = batch[2]
        res = self.forward(x)
        return res, idx

    # ==========================================================
