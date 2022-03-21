import numpy as np
import math
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
  def __init__(self, X_data, y_data) -> None:
    super().__init__()
    self.X_data = np.asarray(X_data)
    self.y_data = np.expand_dims(y_data, axis=1)

  def __len__(self):
    return len(self.y_data)

  def __getitem__(self, idx):
    X_data = torch.as_tensor(self.X_data[idx]).float()
    y_data = torch.as_tensor(self.y_data[idx]).float()

    return X_data, y_data


class Linear_Model(pl.LightningModule):
  def __init__(
      self,
      feature_num: int,
      output_class: int,
      num_layers: int,
      num_units: int
  ) -> None:
    super().__init__()

    module_list = [
        torch.nn.Linear(feature_num, num_units),
        torch.nn.BatchNorm1d(num_features=num_units),
        torch.nn.ReLU()
    ]
    varying_num_units = num_units

    for i in range(num_layers):
      prev_num_units = varying_num_units
      if i >= max(0, num_layers - 2):
        varying_num_units = math.ceil(varying_num_units / 2)

      module_list.extend(
          [torch.nn.Linear(prev_num_units, varying_num_units),
           torch.nn.BatchNorm1d(num_features=varying_num_units),
           torch.nn.ReLU()]
      )

    module_list.extend(
        [torch.nn.Dropout(),
         torch.nn.Linear(varying_num_units, output_class),
         torch.nn.ReLU()]
    )

    self.module_list = torch.nn.ModuleList(module_list)

  def forward(self, x):
    for f in self.module_list:
      x = f(x)
    return x


class BloodQuanPredModel(pl.LightningModule):
  def __init__(
      self,
      param: dict,
      feature_num: int,
      output_class: int
  ) -> None:
    super().__init__()
    self.save_hyperparameters("param")
    self.save_hyperparameters("feature_num")
    self.save_hyperparameters("output_class")

    self.loss = torch.nn.MSELoss()
    self.model = Linear_Model(
        feature_num=feature_num,
        output_class=output_class,
        num_layers=param["num_layers"],
        num_units=param["num_units"]
    )

  def forward(self, x):
      logits = self.model(x)
      return logits

  def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
      return optimizer

  def training_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss = self.loss(logits, y)
      self.log("train_loss", loss)

      return loss

  def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self.forward(x)
      loss = self.loss(logits, y)
      metrics = {"val_loss": loss}
      self.log_dict(metrics, prog_bar=True)

      return {'loss': loss}

  def predict_step(self, batch, batch_idx):
      x, y = batch
      logits = self.forward(x)

      return logits, y
