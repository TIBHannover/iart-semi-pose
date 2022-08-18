import os
import copy
import logging

import torch
import numpy as np
import pytorch_lightning as pl
import torchvision

from pytorch_lightning.utilities.distributed import rank_zero_only

try:
    import wandb
except:
    pass

from utils.plot_utils import plot_prediction_images
from utils.box_ops import boxes_to_abs, box_cxcywh_to_xyxy, point_to_abs


class ProgressPrinter(pl.callbacks.ProgressBarBase):
    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.enabled = True

    @property
    def is_enabled(self):
        return self.enabled and self.refresh_rate > 0

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)
        # print(f"TRAIN_BATCH_END {self.refresh_rate}")
        if self.is_enabled and self.trainer.global_step % self.refresh_rate == 0:
            progress_bar_dict = copy.deepcopy(trainer.progress_bar_dict)

            progress_bar_dict.pop("v_num", None)

            log_parts = []
            for k, v in progress_bar_dict.items():
                if isinstance(v, (float, np.float32)):
                    log_parts.append(f"{k}:{v:.2E}")
                else:
                    log_parts.append(f"{k}:{v}")

            logging.info(f"Train {self.trainer.global_step} " + " ".join(log_parts))

    @rank_zero_only
    def on_validation_end(self, trainer, *args):
        super().on_validation_end(trainer, *args)
        # print("###################")
        # print("VAL_BATCH_END")
        progress_bar_dict = copy.deepcopy(trainer.progress_bar_dict)

        progress_bar_dict.pop("v_num", None)

        log_parts = []
        for k, v in progress_bar_dict.items():

            if isinstance(v, (float, np.float32)):
                log_parts.append(f"{k}:{v:.2E}")
            else:
                log_parts.append(f"{k}:{v}")

        logging.info(f"Val {self.trainer.global_step+1} " + " ".join(log_parts))


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, checkpoint_save_interval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_save_interval = checkpoint_save_interval
        # self.filename = "checkpoint_{global_step}"

    @rank_zero_only
    def on_validation_end(self, trainer, *args):
        # print("SSSSSSSSS")
        self.trainer = trainer
        super().on_validation_end(trainer, *args)

    @rank_zero_only
    def on_train_batch_end(self, trainer, *args, **kwargs):
        super().on_train_batch_end(trainer, *args, **kwargs)
        # print("SAVE")
        if self.checkpoint_save_interval is not None:
            if (trainer.global_step + 1) % self.checkpoint_save_interval == 0:
                # print(f"SAVE {self.checkpoint_save_interval} ")
                self.on_validation_end(trainer, args[0])


class LogModelWeightCallback(pl.callbacks.Callback):
    def __init__(self, log_every_n_steps=None, nrow=2, **kwargs):
        super(LogModelWeightCallback, self).__init__(**kwargs)
        self.log_every_n_steps = log_every_n_steps
        self.nrow = nrow

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        if self.log_every_n_steps is None:
            log_interval = trainer.log_every_n_steps
        else:
            log_interval = self.log_every_n_steps

        if (trainer.global_step + 1) % log_interval == 0:
            for k, v in pl_module.state_dict().items():
                try:
                    trainer.logger.experiment.add_histogram(f"weights/{k}", v, trainer.global_step + 1)
                except ValueError as e:
                    logging.info(f"LogModelWeightCallback: {e}")


class TensorBoardLogImageCallback(pl.callbacks.Callback):
    def __init__(self, log_every_n_steps=None, nrow=2, **kwargs):
        super(TensorBoardLogImageCallback, self).__init__(**kwargs)
        self.log_every_n_steps = log_every_n_steps
        self.nrow = nrow

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

        if trainer.logger is None:
            return

        if not hasattr(trainer.logger.experiment, "add_histogram"):
            logging.warning(f"TensorBoardLogImageCallback: No TensorBoardLogger detected")
            return

        if not hasattr(trainer.logger.experiment, "add_image"):
            logging.warning(f"TensorBoardLogImageCallback: No TensorBoardLogger detected")
            return

        if self.log_every_n_steps is None:
            log_interval = trainer.log_every_n_steps
        else:
            log_interval = self.log_every_n_steps

        if (trainer.global_step + 1) % log_interval == 0:

            grid = torchvision.utils.make_grid(pl_module.image, normalize=True, nrow=self.nrow)
            trainer.logger.experiment.add_image(f"input/image", grid, trainer.global_step + 1)
            try:
                trainer.logger.experiment.add_histogram(f"input/dist", pl_module.image, trainer.global_step + 1)
            except ValueError as e:
                logging.warning(f"TensorBoardLogImageCallback (source/dist): {e}")


class WandbLogImageCallback(pl.callbacks.Callback):
    def __init__(self, log_every_n_steps=None, nrow=2, **kwargs):
        super(WandbLogImageCallback, self).__init__(**kwargs)
        self.log_every_n_steps = log_every_n_steps
        self.nrow = nrow

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

        if trainer.logger is None:
            return

        if self.log_every_n_steps is None:
            log_interval = trainer.log_every_n_steps
        else:
            log_interval = self.log_every_n_steps

        if (trainer.global_step + 1) % log_interval == 0:

            grid = torchvision.utils.make_grid(pl_module.image, normalize=True, nrow=self.nrow)

            trainer.logger.experiment.log(
                {
                    "val/examples": [wandb.Image(grid, caption="Input images")],
                    "global_step": trainer.global_step + 1,
                }
            )


class BoxesImageCallback(pl.callbacks.Callback):
    def __init__(self, log_every_n_steps=None, output_path=None, nrow=2, **kwargs):
        super(BoxesImageCallback, self).__init__(**kwargs)
        self.log_every_n_steps = log_every_n_steps
        self.output_path = output_path
        self.nrow = nrow

    def _log_images(self, step, suffix, image_ids, images, boxes, sizes):
        for i, (image_id, image, boxe, size) in enumerate(zip(image_ids, images.unbind(0), boxes, sizes)):

            image = np.transpose(image.cpu().numpy(), (1, 2, 0))
            boxe = boxes_to_abs(box_cxcywh_to_xyxy(boxe), size)

            image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
            image = (image * 255).astype(np.uint8)
            if not isinstance(image_id, str):
                image_id = image_id.numpy().item()
            plot_prediction_images(image, os.path.join(self.output_path, f"{image_id}_{step}_{suffix}_{i}.png"), boxe)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

        if self.log_every_n_steps is None:
            log_interval = trainer.log_every_n_steps
        else:
            log_interval = self.log_every_n_steps

        if (trainer.global_step + 1) % log_interval == 0:
            if hasattr(pl_module, "boxes_images"):
                for entry in pl_module.boxes_images:
                    self._log_images(
                        trainer.global_step,
                        entry["names"],
                        entry["ids"],
                        entry["images"],
                        entry["boxes"],
                        entry["sizes"],
                    )


class PosesImageCallback(pl.callbacks.Callback):
    def __init__(self, log_every_n_steps=None, output_path=None, nrow=2, **kwargs):
        super(PosesImageCallback, self).__init__(**kwargs)
        self.log_every_n_steps = log_every_n_steps
        self.output_path = output_path
        self.nrow = nrow

    def _log_images(
        self, step, suffix, image_ids, images, keypoints, sizes, keypoints_labels=None, keypoints_scores=None
    ):
        if keypoints_scores is None:
            keypoints_scores = [None for _ in images.unbind(0)]

        if keypoints_labels is None:
            keypoints_labels = [None for _ in images.unbind(0)]

        for i, (image_id, image, keypoint, scores, labels, size) in enumerate(
            zip(image_ids, images.unbind(0), keypoints, keypoints_scores, keypoints_labels, sizes)
        ):
            image = np.transpose(image.cpu().numpy(), (1, 2, 0))
            keypoint = point_to_abs(keypoint, size)

            image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
            image = (image * 255).astype(np.uint8)
            if not isinstance(image_id, str):
                image_id = image_id.numpy().item()

            if labels is not None:
                labels = [labels]
            if scores is not None:
                scores = [scores]
            # print(f"##### {suffix} {labels} {scores}")
            plot_prediction_images(
                image,
                os.path.join(self.output_path, f"{image_id}_{step}_{i}_{suffix}.png"),
                keypoints=[keypoint],
                keypoints_labels=labels,
                keypoints_scores=scores,
            )

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

        if self.log_every_n_steps is None:
            log_interval = trainer.log_every_n_steps
        else:
            log_interval = self.log_every_n_steps

        logging.info(f"PosesImageCallback::on_train_batch_end log_interval:{log_interval}")

        if (trainer.global_step + 1) % log_interval == 0:
            if hasattr(pl_module, "keypoints_images"):
                for entry in pl_module.keypoints_images:
                    # print(entry.keys())
                    # print(entry.get("keypoints"))
                    # print(entry.get("keypoints_labels"))
                    # print(entry.get("keypoints_scores"))
                    # continue
                    self._log_images(
                        trainer.global_step,
                        suffix=entry.get("names"),
                        image_ids=entry.get("ids"),
                        images=entry.get("images"),
                        keypoints=entry.get("keypoints"),
                        keypoints_labels=entry.get("keypoints_labels"),
                        keypoints_scores=entry.get("keypoints_scores"),
                        sizes=entry.get("sizes"),
                    )
                exit()