import os
import sys
import re
import argparse
import logging


from torch import nn


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from callbacks import ModelCheckpoint, ProgressPrinter, LogImageCallback  # , LogImageCallback
from datasets import DatasetsManager
from models import ModelsManager

from pytorch_lightning.utilities.cloud_io import load as pl_load


class ExportWrapper(nn.Module):
    def __init__(self, model, on_gpu=False):
        super(ExportWrapper, self).__init__()
        self.model = model
        self.on_gpu = on_gpu

    def preprocess_input(self, input):
        mean = torch.zeros(3).float().to(input.device)
        std = torch.zeros(3).float().to(input.device)
        mean[0], mean[1], mean[2] = 0.485, 0.456, 0.406
        std[0], std[1], std[2] = 0.229, 0.224, 0.225
        mean = mean.unsqueeze(1).unsqueeze(1)
        std = std.unsqueeze(1).unsqueeze(1)
        temp = input.float().div(255).permute(2, 0, 1).to(input.device)
        return temp.sub(mean).div(std).unsqueeze(0)

    def __call__(self, input):
        if self.on_gpu:
            input = input.to("cuda:0")
        print(input.device)
        if len(input.shape) == 4:
            input_images = []
            for i in range(input.shape[0]):

                input_images.append(self.preprocess_input(input[i, ...]))
            x = torch.stack(input_images, dim=0)
        else:
            x = self.preprocess_input(input)

        image_embedding = self.model.encoder(x)
        image_embedding = image_embedding[0]

        if self.model.use_diverse_beam_search:
            seqs, scores = self.model.diverse_beam_search(
                image_embedding,
                num_groups=self.model.div_beam_s_group,
                diversity_strength=-0.2,
                beam_size=10,
                convert_outputs=False,
            )
            return seqs, scores


def parse_args():
    parser = argparse.ArgumentParser(description="", conflict_handler="resolve")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    parser.add_argument("-d", "--device", help="verbose output")

    parser.add_argument("-o", "--output_file", help="verbose output")

    parser = pl.Trainer.add_argparse_args(parser)
    # parser = DatasetsManager.add_args(parser)
    parser = ModelsManager.add_args(parser)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    level = logging.ERROR
    if args.verbose:
        level = logging.INFO

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    # dataset = DatasetsManager().build_dataset(name=args.dataset, args=args)

    # for x in dataset.test():
    #     print(x)
    # exit()

    model = ModelsManager().build_model(name=args.model, args=args)

    trainer = pl.Trainer.from_argparse_args(args)

    checkpoint_data = torch.load(args.resume_from_checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint_data["state_dict"])
    model.freeze()
    model.eval()

    if args.device == "gpu":
        print("GPU")
        export_model = ExportWrapper(model.to("cuda:0"), on_gpu=True).cuda().eval()
        batch = (torch.ones((224, 224, 3))).to("cuda:0")
    else:
        export_model = ExportWrapper(model.to("cpu"), on_gpu=False).cpu().eval()
        batch = torch.ones((224, 224, 3))

    traced_model = torch.jit.trace(export_model, batch)
    torch.jit.save(traced_model, args.output_file)
    # model.to("cuda:0")

    # for batch in dataset.infer():
    #     # print(batch)
    #     for c in model.infer_step(batch, k=20):
    #         print(f"{c['path']} {c['classes']}")
    #         # print(prediction)

    return 0


if __name__ == "__main__":
    sys.exit(main())
