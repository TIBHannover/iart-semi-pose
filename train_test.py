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
)  # , LogImageCallback
from datasets import DatasetsManager
from models import ModelsManager
from packaging import version

import json

import utils.misc as utils

try:
    import yaml
except:
    yaml = None


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
    parser.add_argument("--progress_refresh_rate", type=int, default=100, help="verbose output")
    parser.add_argument("--wandb_name", help="verbose output")
    parser.add_argument("--checkpoint_save_interval", type=int, default=2000, help="verbose output")
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

        pl.seed_everything(42)

        dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)
        model = ModelsManager().build_model(name=args.model, args=args)
        for x in dataset.train():
            pass
        for x in dataset.val():
            pass
        exit()
        if args.output_path is not None:
            os.makedirs(args.output_path, exist_ok=True)

        callbacks = []
        if get_node_rank() == 0:
            if args.output_path is not None and not args.use_wandb:
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
                logger = WandbLogger(project="iart_pose", log_model=False, name=name)
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

        if version.parse(pl.__version__) < version.parse("1.4"):
            checkpoint_callback = ModelCheckpoint(
                checkpoint_save_interval=args.checkpoint_save_interval,
                dirpath=args.output_path,
                filename="model_{step:06d}",
                save_top_k=-1,
                verbose=True,
                period=1,
            )
        else:
            checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
                dirpath=args.output_path, save_top_k=-1, every_n_train_steps=args.checkpoint_save_interval
            )

        callbacks.extend([checkpoint_callback])

        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            # checkpoint_callback=checkpoint_callback,
        )

        trainer.fit(model, train_dataloader=dataset.train(), val_dataloaders=dataset.val())

        return 0
    except Exception:
        logging.info(traceback.format_exc())
        # or
        logging.info(sys.exc_info()[2])


if __name__ == "__main__":
    sys.exit(main())
