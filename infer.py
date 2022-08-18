import os
import sys
import re
import argparse
import logging
import json

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from datasets import DatasetsManager
from models import ModelsManager

# TODO find a better way to do that
from models.detr import DETRModel
from models.detr_semi import DETRSemiModel
from models.pose_transformer import PoseTransformerModel
from models.pose_transformer_semi import PoseTransformerSemiModel

from utils.misc import detach_all, move_all

from pytorch_lightning.utilities.cloud_io import load as pl_load


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    parser.add_argument("-t", "--threshold", default=0.9, type=float, help="verbose output")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    # for x in dataset.test():
    #     print(x)
    # exit()

    model = ModelsManager().build_model(name=args.model, args=args)

    trainer = pl.Trainer.from_argparse_args(args)
    print(args.resume_from_checkpoint)
    checkpoint_data = pl_load(args.resume_from_checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint_data["state_dict"])
    model.freeze()
    model.eval()
    model.to("cuda:0")
    count = 0
    if isinstance(model, (PoseTransformerModel, PoseTransformerSemiModel)):
        with open(args.output_path, "w") as f:
            for batch in dataset.infer():
                batch = move_all(batch, "cuda:0")
                # print(batch["image"].tensors.shape)
                # print(batch["image"].tensors.device)
                # batch["image"] = batch["image"].to("cuda:0")
                # batch["target"]["transformation"] = batch["target"]["transformation"].to("cuda:0")
                predictions = model.infer_step(batch, args.threshold)
                # print(predictions)
                # exit()
                for id, keypoints, labels, scores, selected, boxes_id, origin_size, size in zip(
                    predictions["image_id"],
                    predictions["keypoints"],
                    predictions["labels"],
                    predictions["scores"],
                    predictions["selected"],
                    predictions["boxes_id"],
                    predictions["origin_size"],
                    predictions["size"],
                ):
                    f.write(
                        json.dumps(
                            {
                                "id": id,
                                "keypoints": [keypoints.numpy().tolist()],
                                "labels": [labels.numpy().tolist()],
                                "scores": [scores.numpy().tolist()],
                                "selected": [selected.numpy().tolist()],
                                "boxes_id": [boxes_id],
                                "origin_size": [origin_size.detach().cpu().numpy().tolist()],
                                "size": [size.detach().cpu().numpy().tolist()],
                            }
                        )
                        + "\n"
                    )
                # print(prediction)
                count += 1
                print(count)
    elif isinstance(model, (DETRModel, DETRSemiModel)):
        with open(args.output_path, "w") as f:
            for batch in dataset.infer():
                batch = move_all(batch, "cuda:0")
                # batch["image"] = batch["image"].to("cuda:0")
                # batch["target"]["transformation"] = batch["target"]["transformation"].to("cuda:0")
                predictions = model.infer_step(batch, args.threshold)
                for id, boxes, labels, scores in zip(
                    predictions["image_id"], predictions["boxes"], predictions["labels"], predictions["scores"]
                ):
                    if isinstance(id, torch.Tensor):
                        id = id.cpu().numpy().item()
                    print(
                        {
                            "id": id,
                            "boxes": boxes.numpy().tolist(),
                            "labels": labels.numpy().tolist(),
                            "scores": scores.numpy().tolist(),
                        }
                    )
                    f.write(
                        json.dumps(
                            {
                                "id": id,
                                "boxes": boxes.numpy().tolist(),
                                "labels": labels.numpy().tolist(),
                                "scores": scores.numpy().tolist(),
                            }
                        )
                        + "\n"
                    )
                # print(prediction)
                count += 1
                print(count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
