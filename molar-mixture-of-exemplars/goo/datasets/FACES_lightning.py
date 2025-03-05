
from lightning import LightningDataModule
from .base_lightning_module import BaseDataModule
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
import numpy as np
from loguru import logger
from PIL import Image

from torch.utils.data import Subset



class FaceDataset(Dataset):
    # https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv
    def __init__(
        self,
        df,
        transform=None,
    ):
        self.df = df.to_dict("records")

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]
        label = row["emotion"]
        pixels = row[" pixels"]

        pixels = np.array(pixels.split()).astype(int)
        pixels = np.reshape(pixels, (48, 48))
        pixels = np.expand_dims(pixels, axis=-1)
        image = np.repeat(pixels, 3, axis=-1)
        image = np.uint8(image)
        image = Image.fromarray(image)


        if self.transform:
            image = self.transform(image)

        return image, label


class OODDataset(Dataset):
    def __init__(
        self, dataset_name='icmlface', split="Out", transform=None, random_state=42, rotnet=False
    ):
        """
        Returns a datasets split based on 75% 25%.

        :param dataset_name:
        :param split:
        :param transform:
        """

        assert split in ["Train", "Test", "Out"]

        self.rotnet = rotnet
        self.dataset_name = dataset_name
        self.return_grouped_cifar = False

        datasource, dataframe = self.build(dataset_name)

        self.dataframe = dataframe
        df = dataframe

        self.in_distro = (
            pd.Series(df.label.unique())
            .sample(frac=0.75, random_state=random_state)
            .values
        )

        self.out_distro = pd.Series(df.label.unique())[
            ~pd.Series(df.label.unique()).isin(self.in_distro)
        ].values

        new_label_int = {
            old: new_label_int for new_label_int, old in enumerate(self.in_distro)
        }  # labels 1 5 6 9 to 0 1 2 3
        new_label_int = None if self.return_grouped_cifar else new_label_int

        if split == "Train":
            self.df = df[
                (df.split == "train") & (df.label.isin(self.in_distro))
            ].reset_index()
            self.source = datasource["train"]
        elif split == "Test":
            self.df = df[
                (df.split == "val") & (df.label.isin(self.in_distro))
            ].reset_index()
            self.source = datasource["validation"]
        elif split == "Out":
            self.df = df[
                (df.split == "val") & (~df.label.isin(self.in_distro))
            ].reset_index()
            self.source = datasource["validation"]

        self.transform = transform
        self.new_label_int = None if split == "Out" else new_label_int

        if self.new_label_int is not None:
            self.targets = [self.new_label_int[i] for i in  list(self.df.label.values)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = row["label"]
        if self.new_label_int is not None:
            label = self.new_label_int[label]
        elif self.return_grouped_cifar:
            label = row["group_idx"]

        data = self.source[int(row["index"])]

        if type(data) is dict:  # TFDS dict style
            image = data["image"]
        else:
            image = data[0]  # Pytorch tuple style

        if type(image) is not Image.Image:
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def build(self, dataset_name):
        if dataset_name == "icmlface":

            df = pd.read_csv(Path(__file__).parent.parent.parent.parent / "data/icml_face_data.csv")

            df["split"] = "unknown"

            df.loc[df[" Usage"] == "Training", "split"] = "train"
            df.loc[df[" Usage"].str.contains("Test"), "split"] = "val"
            df["label"] = df["emotion"]
            df["index"] = df.index
            dataset = FaceDataset(df)
            datasource = {"train": dataset, "validation": dataset}
            dataframe = df


        else:
            raise KeyError(f"Dataset name {dataset_name} is not valid")

        return datasource, dataframe

class FaceDataModule(BaseDataModule):
    def __init__(
            self,
            data_dir = "data/",
            seed = 0,
            **kwargs):
        super().__init__(**kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.seed = seed



    @property
    def num_classes(self):
        return 10

    def setup(self, stage = None):

        # OODDataset()

        dataset_name =  "icmlface"

        train_set = OODDataset(
            dataset_name,
            split="Train",
            transform=None,
            random_state=self.seed,
        )
        test_set = OODDataset(
            dataset_name, split="Test", transform=None, random_state=self.seed
        )
        ood_set = OODDataset(
            dataset_name, split="Out", transform=None, random_state=self.seed
        )


        self.data_train = train_set
        self.data_val = test_set
        self.data_test = test_set
        self.data_ood = ood_set



        # =======================================================
        super().setup(stage)


