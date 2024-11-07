from argparse import ArgumentParser
from typing import Type

import torch
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer

from model.activations import get_class
from model.lightnings import LitBaseModel
from data.datamodule import DataModule


def main(config: DictConfig):
    lightning.seed_everything(config.trainer.get("seed", 2024))
    datamodule = DataModule(config.dataset)

    model_class: Type[LitBaseModel] = get_class(config.lightning.name)

    if config.trainer.get("resume_ckpt_path", None) is not None:
        model: LitBaseModel = model_class.load_from_checkpoint(
            config.trainer.resume_ckpt_path
        )
    else:
        model: LitBaseModel = model_class(config)

    logger = TensorBoardLogger(**config.logger)
    accelerator = (
        "gpu"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "auto"
    )
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = Trainer(
        accelerator,
        devices=1,
        precision=precision,
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/train-brain-model.yml", type=str)
    config = OmegaConf.load(parser.parse_args().config)
    main(config)
