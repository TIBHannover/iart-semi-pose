import os
import sys
import re
import argparse
import logging

import uuid

import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune import CLIReporter

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch

from callbacks import ProgressPrinter
from datasets import DatasetsManager
from models import ModelsManager
from packaging import version

# solve some issues with slurm
os.environ["SLURM_JOB_NAME"] = "bash"


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    # set logging
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_known_args(["-v", "--verbose"])
    level = logging.ERROR
    if args[0].verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    parser.add_argument("--output_path", help="verbose output")
    parser.add_argument("--use_wandb", action="store_true", help="verbose output")
    parser.add_argument("--use_tensorboard", action="store_true", help="verbose output")
    parser.add_argument("--progress_refresh_rate", type=int, default=100, help="verbose output")
    parser.add_argument("--wandb_name", help="verbose output")
    parser.add_argument("--wandb_project", default="iart_pose", help="verbose output")
    parser.add_argument("--checkpoint_save_interval", type=int, default=100, help="verbose output")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)
    args = parser.parse_args()

    return args


def train_tune(config, num_steps=200000, num_gpus=-1):
    args = config["args"]
    del config["args"]

    if "transformer_d_model" in config:
        if hasattr(args, "resnet_output_depth"):
            config["resnet_output_depth"] = config["transformer_d_model"]

    if "decoder_embedding_dim" in config:
        if hasattr(args, "resnet_output_depth"):
            config["resnet_output_depth"] = config["decoder_embedding_dim"]
    for k, v in config.items():
        setattr(args, k, v)
    level = logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    # args.lr = config["lr"]

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    model = ModelsManager().build_model(name=args.model, args=args)

    callbacks = []
    if tune.get_trial_dir() is not None and args.use_tensorboard:
        logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="summary")

    elif args.use_wandb:
        name = f"{args.model}"
        if hasattr(args, "encoder"):
            name += f"-{args.encoder}"
        if hasattr(args, "decoder"):
            name += f"-{args.decoder}"
        name += f"-{uuid.uuid4().hex[:4]}"

        if args.wandb_name is not None:
            name = args.wandb_name
        logger = WandbLogger(project=args.wandb_project, log_model=False, name=tune.get_trial_name())
        logger.watch(model)
        # callbacks.extend([WandbLogImageCallback()])
    else:
        logging.warning("No logger available")
        logger = None

    callbacks += [
        TuneReportCallback({"loss": "val/loss", "map": "val/map"}, on="validation_end"),
        # ProgressPrinter(refresh_rate=1000),
    ]

    callbacks += [
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=tune.get_trial_dir(), save_top_k=3, monitor="val/map", every_n_train_steps=int(num_steps / 50)
        )
    ]

    trainer = pl.Trainer(
        max_steps=num_steps,
        # If fractional GPUs passed in, convert to int.
        gpus=num_gpus,
        logger=logger,
        progress_bar_refresh_rate=0,
        callbacks=callbacks,
        val_check_interval=int(num_steps / 50),
        precision=16,
        num_sanity_val_steps=0,
    )
    trainer.running_sanity_check = False
    trainer.fit(model, train_dataloaders=dataset.train(), val_dataloaders=dataset.val())


def main():
    args = parse_args()

    # ray.init(num_gpus=1)

    pl.seed_everything(42)

    config_model = ModelsManager.tunning_scopes(args)
    config_dataset = DatasetsManager.tunning_scopes(args)
    config = {**config_dataset, **config_model}
    print(config_model)
    print(config_dataset)

    config["args"] = args

    scheduler = tune.schedulers.ASHAScheduler(
        metric="map",
        mode="max",
    )
    analysis = tune.run(
        train_tune,
        config=config,
        num_samples=100,
        scheduler=scheduler,
        name="tune_pose_poses",
        resources_per_trial={"gpu": 1},
    )

    best_trial = analysis.get_best_trial("map", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation mao: {}".format(best_trial.last_result["map"]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
