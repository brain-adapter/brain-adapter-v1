from argparse import ArgumentParser

import torch
import lightning
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer

from model.activations import get_class
from model.lightnings import LitBaseModel
from data.datamodule import DataModule


def main(config: DictConfig):
    lightning.seed_everything(config.trainer.get("seed", 2024))
    datamodule = DataModule(config.dataset)

    model_class: LitBaseModel = get_class(config.lightning.name)

    model = model_class(config)

    logger = TensorBoardLogger(**config.logger)
    accelerator = (
        "gpu"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "auto"
    )
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    # load callbacks
    early_stop_call_back = (
        plc.EarlyStopping(**config.trainer.early_stop)
        if config.trainer.get("early_stop", None) is not None
        else None
    )
    lr_monitor_call_back = plc.LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = (
        plc.ModelCheckpoint(**config.trainer.checkpoint)
        if config.trainer.get("checkpoint", None) is not None
        else None
    )

    callbacks = [
        item
        for item in (early_stop_call_back, lr_monitor_call_back, ckpt_callback)
        if item is not None
    ]

    trainer = Trainer(
        accelerator,
        devices=config.trainer.get("num_devices", 1),
        precision=precision,
        accumulate_grad_batches=config.trainer.get("accumulate_grad_batches", 1),
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.trainer.get("max_epochs", None),
        max_steps=config.trainer.get("max_steps", -1),
        val_check_interval=config.trainer.get("val_check_interval", None),
        check_val_every_n_epoch=config.trainer.get("check_val_every_n_epoch", None),
        enable_checkpointing=ckpt_callback is not None,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=config.trainer.get("resume_ckpt_path", None))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/train-brain-model.yml", type=str)
    config = OmegaConf.load(parser.parse_args().config)
    main(config)
