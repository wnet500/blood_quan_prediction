import pandas as pd
import pytest
import torch

from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from main.reference.modeling import ModelTrainer
from main.reference.prepocessing import DataProcessor
from main.reference.torch_modeling_reference import CustomDataset
from main.reference.utils import adjust_pred_value, get_adjusted_r2, disable_logging_and_userwaring
from tests import __train_data_file__ as train
from tests import __test_data_file__ as test


@pytest.fixture
def data_processor():
  return DataProcessor(train, test)


@pytest.fixture
def X_y_datasets(data_processor):
  X_trainval, X_test, y_trainval, y_test = data_processor.make_X_y_datasets()
  data_dict = {
      "X_trainval": X_trainval, "X_test": X_test,
      "y_trainval": y_trainval, "y_test": y_test
  }
  return data_dict


@pytest.fixture
def model_trainer(X_y_datasets):
  return ModelTrainer(X_y_datasets["X_trainval"], X_y_datasets["y_trainval"])


def load_gridsearch_result(filename):
  dir_path = Path(__file__).parent.parent.joinpath("output", "gridsearch_results")
  return pd.read_csv(str(dir_path.joinpath(filename)))


def test_linear_regression_evaluation(X_y_datasets, model_trainer):
  X_test, y_test = X_y_datasets["X_test"], X_y_datasets["y_test"]
  lr_model = model_trainer.train_linear_regression()
  y_pred = adjust_pred_value(lr_model.predict(X_test))
  print()
  print(f"lr_mse: {mean_squared_error(y_test, y_pred):.3f}")
  print(f"lr_adj_r2: {get_adjusted_r2(y_test, y_pred, X_test.shape[1]):.3f}")


def test_random_forest_evaluation(X_y_datasets, model_trainer):
  best_param = \
      load_gridsearch_result("rf_results.csv")\
      .sort_values(by="rf_mse_mean")\
      .head(1)["param"].values[0]

  X_test, y_test = X_y_datasets["X_test"], X_y_datasets["y_test"]
  rf_model = model_trainer.train_random_forest(eval(best_param))
  y_pred = adjust_pred_value(rf_model.predict(X_test))
  print()
  print(f"rf_mse: {mean_squared_error(y_test, y_pred):.3f}")
  print(f"rf_adj_r2: {get_adjusted_r2(y_test, y_pred, X_test.shape[1]):.3f}")


def test_xgb_evaluation(X_y_datasets, model_trainer):
  best_param, n_estimators = \
      load_gridsearch_result("xgb_results.csv")\
      .sort_values(by="xgb_mse_mean")\
      .head(1)[["param", "early_stopping_round_mean"]].values[0]

  X_test, y_test = X_y_datasets["X_test"], X_y_datasets["y_test"]
  tree_method = "auto"  # "gpu_hist"
  xgb_model = model_trainer.train_xgboost(param=eval(
      best_param), tree_method=tree_method, n_estimators=int(n_estimators))
  y_pred = adjust_pred_value(xgb_model.predict(X_test))
  print()
  print(f"xgb_mse: {mean_squared_error(y_test, y_pred):.3f}")
  print(f"xgb_adj_r2: {get_adjusted_r2(y_test, y_pred, X_test.shape[1]):.3f}")


def test_ann_evaluation(X_y_datasets, model_trainer):
  best_param, epoch = \
      load_gridsearch_result("ann_results.csv")\
      .sort_values(by="ann_mse_mean")\
      .head(1)[["param", "best_epoch_mean"]].values[0]

  X_test, y_test = X_y_datasets["X_test"], X_y_datasets["y_test"]
  test_dataset = CustomDataset(X_test, y_test)
  testset_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)

  disable_logging_and_userwaring()

  ann_trainer, ann_model = model_trainer.train_ann(
      param=eval(best_param),
      num_epochs=int(epoch),
      has_bar_callback=False,
      save_model_file="final_ann_model"
  )
  results = ann_trainer.predict(ann_model, dataloaders=testset_loader)
  y_pred = torch.vstack([results[i][0] for i in range(len(results))]).cpu().numpy()[:, 0]
  print()
  print(f"ann_mse: {mean_squared_error(y_test, adjust_pred_value(y_pred)):.3f}")
  print(f"ann_adj_r2: {get_adjusted_r2(y_test, adjust_pred_value(y_pred), X_test.shape[1]):.3f}")


def test_prev_practice_evaluation(data_processor):
  _, prev_practice_test, _, y_test = data_processor.make_prev_practice_datasets()

  print()
  print(f"msbos_mse: {mean_squared_error(y_test, prev_practice_test):.3f}")
  print(f"msbos_r2: {r2_score(y_test, prev_practice_test):.3f}")
