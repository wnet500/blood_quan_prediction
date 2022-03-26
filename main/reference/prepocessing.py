import json
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List


class DataProcessor:
  def __init__(self, train_data_file: str, test_data_file: str) -> None:
      self.data_dir = Path(__file__).parent.parent.parent.joinpath("data")
      self.train_data_file = train_data_file
      self.test_data_file = test_data_file
      self.datasets_info = self._get_datasets_info()

  def _get_datasets_info(self):
    with open(str(self.data_dir.joinpath("datasets_info.json")), "r") as fp:
      datasets_info = json.load(fp)

    return datasets_info

  def load_raw_datasets(self) -> List[pd.DataFrame]:
    train_data = pd.read_csv(str(self.data_dir.joinpath(self.train_data_file)))
    test_data = pd.read_csv(str(self.data_dir.joinpath(self.test_data_file)))

    return [train_data, test_data]

  def convert_into_dummy_coded_datasets(
      self,
      df: pd.DataFrame,
      one_hot_vars: List[str],
      float_vars: List[str]
  ) -> pd.DataFrame:
    if one_hot_vars:
      df[one_hot_vars] = df[one_hot_vars].astype(object)
    if float_vars:
      df[float_vars] = df[float_vars].astype(float)
    df = pd.get_dummies(df)

    return df

  def make_ml_datasets(self) -> List[pd.DataFrame]:
    train_data, test_data = self.load_raw_datasets()

    datasets = pd.concat([train_data.assign(train_data_yn=1), test_data.assign(train_data_yn=0)])
    datasets = datasets.drop(columns=self.datasets_info["drop_columns"])

    dummy_datasets = self.convert_into_dummy_coded_datasets(
        df=datasets,
        one_hot_vars=self.datasets_info["preprocessing"]["one_hot_columns"],
        float_vars=self.datasets_info["preprocessing"]["float_columns"]
    )

    train_data = dummy_datasets.query("train_data_yn == 1").drop(columns=["train_data_yn"])
    test_data = dummy_datasets.query("train_data_yn == 0").drop(columns=["train_data_yn"])

    return [train_data, test_data]

  def make_prev_practice_datasets(self) -> List[np.ndarray]:
    train_data, test_data = self.load_raw_datasets()

    prev_practice_trainval = train_data.filter(
        items=self.datasets_info["prev_practice_column"],
        axis="columns"
    ).squeeze()
    y_trainval = train_data.filter(
        items=self.datasets_info["outcome_columns"],
        axis="columns"
    ).squeeze()

    prev_practice_test = test_data.filter(
        items=self.datasets_info["prev_practice_column"],
        axis="columns"
    ).squeeze()
    y_test = test_data.filter(
        items=self.datasets_info["outcome_columns"],
        axis="columns"
    ).squeeze()

    return [prev_practice_trainval.values, prev_practice_test.values, y_trainval.values, y_test.values]

  def make_X_y_datasets(self) -> List[np.ndarray]:
    trainval_datasets, test_datasets = self.make_ml_datasets()
    X_trainval = trainval_datasets.drop(columns=self.datasets_info["outcome_columns"])
    y_trainval = trainval_datasets.filter(
        items=self.datasets_info["outcome_columns"],
        axis="columns"
    ).squeeze()
    X_test = test_datasets.drop(columns=self.datasets_info["outcome_columns"])
    y_test = test_datasets.filter(
        items=self.datasets_info["outcome_columns"],
        axis="columns"
    ).squeeze()

    return [X_trainval.values, X_test.values, y_trainval.values, y_test.values]
