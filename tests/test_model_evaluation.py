import pandas as pd
import pytest

from pathlib import Path
from sklearn.metrics import mean_squared_error

from main.reference.modeling import ModelTrainer
from main.reference.prepocessing import DataProcessor
from main.reference.utils import adjust_pred_value, get_adjusted_r2
from tests import __train_data_file__ as train
from tests import __test_data_file__ as test


@pytest.fixture
def X_y_datasets():
  data_processor = DataProcessor(train, test)
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
  xgb_model = model_trainer.train_xgboost(param=eval(best_param), tree_method=tree_method, n_estimators=int(n_estimators))
  y_pred = adjust_pred_value(xgb_model.predict(X_test))
  print()
  print(f"xgb_mse: {mean_squared_error(y_test, y_pred):.3f}")
  print(f"xgb_adj_r2: {get_adjusted_r2(y_test, y_pred, X_test.shape[1]):.3f}")
  