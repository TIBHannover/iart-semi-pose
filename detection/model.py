import pytorch_lightning as pl

from transformers import DetrModel, DetrConfig


class DETRDetection(pl.LightningModule):
    def __init__(self):
        super(DETRDetection, self).__init__()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_nb):
        loss = 0.0
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)