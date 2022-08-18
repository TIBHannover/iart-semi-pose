import os
import sys
import re
import argparse
import logging

import uuid


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch

from callbacks import (
    ModelCheckpoint,
    ProgressPrinter,
    TensorBoardLogImageCallback,
    WandbLogImageCallback,
    BoxesImageCallback,
    PosesImageCallback,
)
from datasets import DatasetsManager
from models import ModelsManager
from packaging import version

import json

import utils.misc as utils

try:
    import yaml
except:
    yaml = None

from pytorch_lightning.utilities.cloud_io import load as pl_load


def get_node_rank():
    # if not torch.distributed.is_initialized():

    node_rank = os.environ.get("LOCAL_RANK")
    if node_rank is not None:
        return node_rank

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    # set logging
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_known_args(["-v", "--verbose"])
    level = logging.ERROR
    if args[0].verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    # parse config
    parser.add_argument("-c", "--config_path", help="verbose output")

    args = parser.parse_known_args()
    if args[0].config_path:
        if re.match("^.*?\.(yml|yaml)$", args[0].config_path):
            with open(args[0].config_path, "r") as f:
                data_dict = yaml.safe_load(f)
                parser.set_defaults(**data_dict)

        if re.match("^.*?\.(json)$", args[0].config_path):
            with open(args[0].config_path, "r") as f:
                data_dict = json.load(f)
                parser.set_defaults(**data_dict)

    # add arguments
    parser.add_argument("--output_path", help="verbose output")
    parser.add_argument("--use_wandb", action="store_true", help="verbose output")
    parser.add_argument("--use_tensorboard", action="store_true", help="verbose output")
    parser.add_argument("--progress_refresh_rate", type=int, default=100, help="verbose output")
    parser.add_argument("--wandb_name", help="verbose output")
    parser.add_argument("--wandb_project", default="iart_pose", help="verbose output")
    parser.add_argument("--seed", type=int, default=42, help="verbose output")
    parser.add_argument("--log_boxes_images", action="store_true")
    parser.add_argument("--log_poses_images", action="store_true")

    parser.add_argument("--load_from_checkpoint")

    parser.add_argument("--checkpoint_save_interval", type=int, default=10000, help="verbose output")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)
    args = parser.parse_args()

    # write results

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        if yaml is not None:
            with open(os.path.join(args.output_path, "config.yaml"), "w") as f:
                yaml.dump(vars(args), f, indent=4)

        # with open(os.path.join(args.output_path, "config.json"), "w") as f:
        #     json.dump(vars(args), f, indent=4)

    return args


def main():
    import traceback
    import sys

    try:
        args = parse_args()

        pl.seed_everything(args.seed)

        dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)
        model = ModelsManager().build_model(name=args.model, args=args)

        if args.load_from_checkpoint:

            checkpoint_data = pl_load(args.load_from_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint_data["state_dict"], strict=False)

        if args.output_path is not None:
            os.makedirs(args.output_path, exist_ok=True)

        callbacks = []
        if get_node_rank() == 0:
            if args.output_path is not None and args.use_tensorboard:
                logger = TensorBoardLogger(save_dir=args.output_path, name="summary")

                callbacks.extend([TensorBoardLogImageCallback])
            elif args.use_wandb:
                name = f"{args.model}"
                if hasattr(args, "encoder"):
                    name += f"-{args.encoder}"
                if hasattr(args, "decoder"):
                    name += f"-{args.decoder}"
                name += f"-{uuid.uuid4().hex[:4]}"

                if args.wandb_name is not None:
                    name = args.wandb_name
                logger = WandbLogger(project=args.wandb_project, log_model=False, name=name)
                logger.watch(model)
                # callbacks.extend([WandbLogImageCallback()])
            else:
                logging.warning("No logger available")
                logger = None
        else:
            logger = None

        if logger is not None:
            callbacks += [
                ProgressPrinter(refresh_rate=args.progress_refresh_rate),
                pl.callbacks.LearningRateMonitor(),
            ]

        if args.log_boxes_images:
            callbacks += [BoxesImageCallback(output_path=args.output_path)]

        if args.log_poses_images:
            callbacks += [PosesImageCallback(output_path=args.output_path)]

        callbacks += [
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                dirpath=args.output_path, save_top_k=-1, every_n_train_steps=args.checkpoint_save_interval
            )
        ]

        logging.info(args)

        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            # checkpoint_callback=checkpoint_callback,
        )

        trainer.fit(model, train_dataloaders=dataset.train(), val_dataloaders=dataset.val())

        return 0
    except Exception:
        logging.info(traceback.format_exc())
        # or
        logging.info(sys.exc_info()[2])


if __name__ == "__main__":
    sys.exit(main())
