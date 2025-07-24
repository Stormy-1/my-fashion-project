import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model

class CelebADataModule(pl.LightningDataModule):
    def __init__(self, data_dir, img_size=128, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    
    def setup(self, stage=None):
        self.train_ds = CelebA(self.data_dir, split="train", target_type="attr",
                               download=True, transform=self.transform)
        self.val_ds = CelebA(self.data_dir, split="valid", target_type="attr",
                             download=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

class CelebAClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        # e.g., 40 attributes: smiling, eyeglasses, etc.
        self.backbone = create_model("resnet50", pretrained=True, num_classes=0)
        nf = self.backbone.num_features
        self.classifier = nn.Linear(nf, 40)
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def common_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        preds = torch.sigmoid(logits)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        self.common_step(batch, "val")
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
    data_dir = "data/celeba"
    dm = CelebADataModule(data_dir)
    model = CelebAClassifier()

    trainer = pl.Trainer(
        max_epochs=5,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=20
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
