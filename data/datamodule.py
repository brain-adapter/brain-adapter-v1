from typing import Type
import lightning
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from model.activations import get_class


class DataModule(lightning.LightningDataModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_workers: int = config.get("num_workers", 4)
        self.dataset_class: Type = get_class(config.name)
        self.batch_size: int = config.batch_size

        self.config = config

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.trainset = self.dataset_class(mode="train", config=self.config)
            self.valset = self.dataset_class(mode="val", config=self.config)

        if stage == "test" or stage is None:
            self.testset = self.dataset_class(mode="test", config=self.config)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            drop_last=False,
        )
