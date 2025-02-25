import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from main import train
from tests.helpers.run_if import RunIf
from tests.helpers import helpers 
from tests.helpers import seetests

from pdb import set_trace as pb
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np

from torch import tensor
import torch
nan = float('nan')

# ===========================================================================
# python main.py +run=base.yaml +R=paws-orig +model/dataset=cifar10 +model/networks=dinov2_vits_14_paws_encoder  +debug=debug

# pytest -s tests/test_train.py::test_paws_orig_fast_dev_run_gpu
@RunIf(min_gpus=1)
def test_paws_orig_fast_dev_run_gpu(cfg_paws_orig_cifar10):
    prev_results_train = {'lr-LARS/pg1': tensor(5.7900), 'lr-LARS/pg1-momentum': tensor(0.9000), 'lr-LARS/pg2': tensor(5.7900), 'lr-LARS/pg2-momentum': tensor(0.9000), 'lr-LARS/pg3': tensor(5.7900), 'lr-LARS/pg3-momentum': tensor(0.9000), 'lr-LARS/pg4': tensor(5.7900), 'lr-LARS/pg4-momentum': tensor(0.9000), 'train/loss': tensor(0.0196), 'train/loss_step': tensor(0.0196), 'val/loss': tensor(0.), 'val/accuracy': tensor(0.5000), 'val/mean_class_accuracy': tensor(0.5000), 'val/aucroc': tensor(0.2500), 'val/average_precision_0': tensor(nan), 'val/average_precision_1': tensor(nan), 'val/average_precision_2': tensor(nan), 'val/average_precision_3': tensor(0.5000), 'val/average_precision_4': tensor(nan), 'val/average_precision_5': tensor(nan), 'val/average_precision_6': tensor(1.), 'val/average_precision_7': tensor(nan), 'val/average_precision_8': tensor(1.), 'val/average_precision_9': tensor(nan), 'val/loss_semi': tensor(0.), 'val/ploss': tensor(0.), 'val/me_max': tensor(0.), 'train/loss_epoch': tensor(0.0196), 'train/loss_semi': tensor(0.0196), 'train/ploss': tensor(2.1682), 'train/me_max': tensor(-2.1486), 'test/loss': tensor(0.), 'test/accuracy': tensor(0.2500), 'test/mean_class_accuracy': tensor(0.2000), 'test/aucroc': tensor(0.2500), 'test/average_precision_0': tensor(nan), 'test/average_precision_1': tensor(nan), 'test/average_precision_2': tensor(nan), 'test/average_precision_3': tensor(nan), 'test/average_precision_4': tensor(0.2500), 'test/average_precision_5': tensor(nan), 'test/average_precision_6': tensor(nan), 'test/average_precision_7': tensor(1.), 'test/average_precision_8': tensor(1.), 'test/average_precision_9': tensor(nan), 'test/loss_semi': tensor(0.), 'test/ploss': tensor(0.), 'test/me_max': tensor(0.)}
    seetests.pretrain_eval_finetune_test(cfg_paws_orig_cifar10, prev_results_train, accelerator='gpu', check_keys_train=False)

# python main.py +run=base.yaml +R=paws-vMF-SNE +model/dataset=cifar10 +model/networks=dinov2_vits_14_paws_encoder  +debug=debug

# pytest -s tests/test_train.py::test_paws_vmf_sne_fast_dev_run_gpu
@RunIf(min_gpus=1)
def test_paws_vmf_sne_fast_dev_run_gpu(cfg_paws_vmf_sne_cifar10):
    prev_results_train = {'lr-LARS/pg1': tensor(5.7900), 'lr-LARS/pg1-momentum': tensor(0.9000), 'lr-LARS/pg2': tensor(5.7900), 'lr-LARS/pg2-momentum': tensor(0.9000), 'lr-LARS/pg3': tensor(5.7900), 'lr-LARS/pg3-momentum': tensor(0.9000), 'lr-LARS/pg4': tensor(5.7900), 'lr-LARS/pg4-momentum': tensor(0.9000), 'train/loss': tensor(0.0332), 'train/loss_step': tensor(0.0332), 'train/loss_epoch': tensor(0.0332), 'train/tsne_loss': tensor(0.0332), 'train/Hdiff_max': tensor(7.3910e-06), 'train/max_tries': tensor(15.), 'train/kappa': tensor(1.5039), 'val/loss': tensor(10.6191), 'val/tsne_loss': tensor(10.6191), 'val/Hdiff_max': tensor(2.0149), 'val/max_tries': tensor(50.), 'val/kappa': tensor(1.3357e-15), 'val/backbone_kNN_accuracy': tensor(0.8750), 'val/backbone_kNN_accuracy_mean': tensor(0.8000), 'val/backbone_kNN_accuracy_mIoU': tensor(0.7000), 'val/backbone_head_semi_kNN_accuracy': tensor(0.8750), 'val/backbone_head_semi_kNN_accuracy_mean': tensor(0.8000), 'val/backbone_head_semi_kNN_accuracy_mIoU': tensor(0.7000), 'test/loss': tensor(10.6503), 'test/tsne_loss': tensor(10.6503), 'test/Hdiff_max': tensor(2.0149), 'test/max_tries': tensor(50.), 'test/kappa': tensor(1.1864e-30), 'test/backbone_kNN_accuracy': tensor(0.8750), 'test/backbone_kNN_accuracy_mean': tensor(0.8000), 'test/backbone_kNN_accuracy_mIoU': tensor(0.7000), 'test/backbone_head_semi_kNN_accuracy': tensor(0.8750), 'test/backbone_head_semi_kNN_accuracy_mean': tensor(0.6667), 'test/backbone_head_semi_kNN_accuracy_mIoU': tensor(0.6667)}
    seetests.pretrain_eval_finetune_test(cfg_paws_vmf_sne_cifar10, prev_results_train, accelerator='gpu', check_keys_train=False)



# python main.py +run=base.yaml +R=molar-SS +model/dataset=cifar10 +model/networks=dinov2_vits_14_load_head +debug=debug

# pytest -s tests/test_train.py::test_molar_ss_fast_dev_run_gpu
@RunIf(min_gpus=1)
def test_molar_ss_fast_dev_run_gpu(cfg_molar_ss_cifar10):
    # prev_results_train = {'lr-LARS/pg1': tensor(5.7900), 'lr-LARS/pg1-momentum': tensor(0.9000), 'lr-LARS/pg2': tensor(5.7900), 'lr-LARS/pg2-momentum': tensor(0.9000), 'lr-LARS/pg3': tensor(5.7900), 'lr-LARS/pg3-momentum': tensor(0.9000), 'lr-LARS/pg4': tensor(5.7900), 'lr-LARS/pg4-momentum': tensor(0.9000), 'train/loss': tensor(0.1505), 'train/loss_step': tensor(0.1505), 'val/loss': tensor(0.), 'val/accuracy': tensor(1.), 'val/mean_class_accuracy': tensor(1.), 'val/aucroc': tensor(0.3000), 'val/average_precision_0': tensor(nan), 'val/average_precision_1': tensor(nan), 'val/average_precision_2': tensor(nan), 'val/average_precision_3': tensor(1.), 'val/average_precision_4': tensor(nan), 'val/average_precision_5': tensor(nan), 'val/average_precision_6': tensor(1.), 'val/average_precision_7': tensor(nan), 'val/average_precision_8': tensor(1.), 'val/average_precision_9': tensor(nan), 'val/loss_semi': tensor(0.), 'val/ploss': tensor(0.), 'val/me_max': tensor(0.), 'train/loss_epoch': tensor(0.1505), 'train/loss_semi': tensor(0.1505), 'train/ploss': tensor(2.0239), 'train/me_max': tensor(-1.8734), 'test/loss': tensor(0.), 'test/accuracy': tensor(0.5000), 'test/mean_class_accuracy': tensor(0.2500), 'test/aucroc': tensor(0.2500), 'test/average_precision_0': tensor(nan), 'test/average_precision_1': tensor(nan), 'test/average_precision_2': tensor(nan), 'test/average_precision_3': tensor(nan), 'test/average_precision_4': tensor(1.), 'test/average_precision_5': tensor(nan), 'test/average_precision_6': tensor(nan), 'test/average_precision_7': tensor(0.2500), 'test/average_precision_8': tensor(1.), 'test/average_precision_9': tensor(nan), 'test/loss_semi': tensor(0.), 'test/ploss': tensor(0.), 'test/me_max': tensor(0.)}
    prev_results_train = {'lr-LARS/pg1': tensor(5.7900), 'lr-LARS/pg1-momentum': tensor(0.9000), 'lr-LARS/pg2': tensor(5.7900), 'lr-LARS/pg2-momentum': tensor(0.9000), 'lr-LARS/pg3': tensor(5.7900), 'lr-LARS/pg3-momentum': tensor(0.9000), 'lr-LARS/pg4': tensor(5.7900), 'lr-LARS/pg4-momentum': tensor(0.9000), 'train/loss': tensor(0.0065), 'train/loss_step': tensor(0.0065), 'val/loss': tensor(0.), 'val/accuracy': tensor(0.2500), 'val/mean_class_accuracy': tensor(0.2500), 'val/aucroc': tensor(0.2000), 'val/average_precision_0': tensor(nan), 'val/average_precision_1': tensor(nan), 'val/average_precision_2': tensor(nan), 'val/average_precision_3': tensor(0.5000), 'val/average_precision_4': tensor(nan), 'val/average_precision_5': tensor(nan), 'val/average_precision_6': tensor(1.), 'val/average_precision_7': tensor(nan), 'val/average_precision_8': tensor(0.2500), 'val/average_precision_9': tensor(nan), 'val/loss_semi': tensor(0.), 'val/ploss': tensor(0.), 'val/me_max': tensor(0.), 'train/loss_epoch': tensor(0.0065), 'train/loss_semi': tensor(0.0065), 'train/ploss': tensor(2.2788), 'train/me_max': tensor(-2.2723), 'test/loss': tensor(0.), 'test/accuracy': tensor(0.), 'test/mean_class_accuracy': tensor(0.), 'test/aucroc': tensor(0.1667), 'test/average_precision_0': tensor(nan), 'test/average_precision_1': tensor(nan), 'test/average_precision_2': tensor(nan), 'test/average_precision_3': tensor(nan), 'test/average_precision_4': tensor(0.2500), 'test/average_precision_5': tensor(nan), 'test/average_precision_6': tensor(nan), 'test/average_precision_7': tensor(0.5000), 'test/average_precision_8': tensor(0.5000), 'test/average_precision_9': tensor(nan), 'test/loss_semi': tensor(0.), 'test/ploss': tensor(0.), 'test/me_max': tensor(0.)}
    seetests.pretrain_eval_finetune_test(cfg_molar_ss_cifar10, prev_results_train, accelerator='gpu', check_keys_train=False)

# python main.py +run=base.yaml +R=molar +model/dataset=cifar10 +model/networks=dinov2_vits_14_load_head +debug=debug

# python main.py +run=base.yaml +R=save-representation +model/dataset=cifar10 +model/networks=dinov2_vits_14_paws_encoder +debug=debug
