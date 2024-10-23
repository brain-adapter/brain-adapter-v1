import os
from argparse import ArgumentParser
from enum import Enum
from typing import Optional
from pathlib2 import Path

import torch
import lightning
import lightning.pytorch.callbacks as plc
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer

from model.activations import get_class
from model.lightnings import LitBaseModel
from data.datamodule import DataModule


class Task(Enum):
    TRAIN_ONLY = 1
    TRAIN_AND_TEST_WITH_BEST = 2
    TRAIN_AND_TEST_WITH_LAST = 3
    TEST_ONLY = 4


def save_model(model: Optional[LitBaseModel], save_directory: Optional[str]):
    if model is None or save_directory is None:
        return
    model.save_pretrained(save_directory)


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
        # strategy="ddp_find_unused_parameters_true",
    )

    # resolve task
    task_name: str = config.trainer.get("task", Task.TRAIN_AND_TEST_WITH_BEST.name)
    task = Task[task_name.upper()]

    if task != Task.TEST_ONLY:
        trainer.fit(model, datamodule=datamodule)

    if task == Task.TRAIN_AND_TEST_WITH_BEST and os.path.exists(
        ckpt_callback.best_model_path
    ):
        model = model_class.load_from_checkpoint(ckpt_callback.best_model_path)
    elif task == Task.TRAIN_AND_TEST_WITH_LAST and os.path.exists(
        ckpt_callback.last_model_path
    ):
        model = model_class.load_from_checkpoint(ckpt_callback.last_model_path)
    elif task == Task.TRAIN_ONLY:
        model = None

    if task != Task.TRAIN_ONLY and model is not None:
        trainer.test(model, datamodule=datamodule)

    # save model
    save_directory = config.trainer.get("save_directory", None)

    if (
        ckpt_callback is not None
        and task != Task.TEST_ONLY
        and trainer.is_global_zero
    ):
        ckpt_paths = ckpt_callback.best_k_models.keys()

        for path in ckpt_paths:
            model = model_class.load_from_checkpoint(path, map_location="cpu")
            monitor = Path(path).stem.split("-")[1]
            # save weights and config only
            if save_directory is not None:
                save_model(
                    model,
                    os.path.join(
                        save_directory, "-".join([config.logger.name, monitor])
                    ),
                )
            # remove lightning checkpoints
            os.remove(path)
        if os.path.exists(ckpt_callback.last_model_path):
            if save_directory is not None:
                save_model(
                    model,
                    os.path.join(
                        save_directory, "-".join([config.logger.name, "last"])
                    ),
                )
            os.remove(ckpt_callback.last_model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default="configs/train-brain-model.yml", type=str
    )
    config = OmegaConf.load(parser.parse_args().config)
    main(config)
