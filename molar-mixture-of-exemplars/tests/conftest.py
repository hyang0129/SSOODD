"""This file prepares config fixtures for other tests."""

import pytest
from tests.helpers import helpers 
from pdb import set_trace as pb
from omegaconf import DictConfig, open_dict
from pathlib import Path

@pytest.fixture(scope="function")
def cfg_paws_orig_cifar10(tmp_path) -> DictConfig:
    cfg = {}
    cfg['pretrain'] = helpers.load_configuration_file(Path.joinpath(tmp_path,'pretrain'), 
        override = ["R=paws-orig", "model/dataset=cifar10", "model/networks=dinov2_vits_14_paws_encoder"])
    return cfg

@pytest.fixture(scope="function")
def cfg_paws_vmf_sne_cifar10(tmp_path) -> DictConfig:
    cfg = {}
    cfg['pretrain'] = helpers.load_configuration_file(Path.joinpath(tmp_path,'pretrain'), 
        override = ["R=paws-vMF-SNE", "model/dataset=cifar10", "model/networks=dinov2_vits_14_paws_encoder"])
    return cfg

@pytest.fixture(scope="function")
def cfg_molar_ss_cifar10(tmp_path) -> DictConfig:
    cfg = {}
    cfg['pretrain'] = helpers.load_configuration_file(Path.joinpath(tmp_path,'pretrain'), 
        override = ["R=molar-SS", "model/dataset=cifar10", "model/networks=dinov2_vits_14_load_head"])
    return cfg

