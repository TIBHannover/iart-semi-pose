import os
import sys
import re
import argparse
import logging


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from callbacks import ModelCheckpoint, ProgressPrinter
from datasets import DatasetsManager
from models import ModelsManager

from pytorch_lightning.utilities.cloud_io import load as pl_load


import json

try:
    import yaml
except:
    yaml = None


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
    parser.add_argument("--progress_refresh_rate", type=int, default=100, help="verbose output")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    model = ModelsManager().build_model(name=args.model, args=args)

    trainer = pl.Trainer.from_argparse_args(args)

    checkpoint_data = pl_load(args.resume_from_checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint_data["state_dict"])
    model.freeze()
    model.eval()

    results = trainer.test(model, test_dataloaders=dataset.test())
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(json.dumps(results[0], indent=2))
    else:
        print(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
