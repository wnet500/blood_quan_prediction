import pytest

from sklearn.model_selection import RepeatedKFold

from main.gridsearch_cv import ParamGridSearch
from tests import __train_data_file__ as train
from tests import __test_data_file__ as test

CV_N_SPLITS = 5
CV_N_REPEATS = 1


@pytest.fixture
def param_search():
  cv = RepeatedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=0)
  return ParamGridSearch(train, test, cv)


def test_conduct_lr_cv(param_search):
  param_search.conduct_lr_cv()


def test_conduct_rf_cv(param_search):
  grid_params = {
      'bootstrap': [False],
      'max_depth': [10],
      'max_features': ['sqrt'],
      'min_samples_leaf': [1],
      'min_samples_split': [10],
      'n_estimators': [100, 500]
  }
  param_search.conduct_rf_cv(grid_params)


def test_conduct_xgb_cv(param_search):
  grid_params = {
      'colsample_bytree': [1],
      'gamma': [0.1],
      'learning_rate': [0.01, 0.1],
      'max_depth': [5, 7],
      'reg_lambda': [2],
      'subsample': [0.8]
  }
  tree_method = "auto"  # "gpu_hist"
  valid_size_in_whole_datasets = 0.1
  valid_size_in_trainval = valid_size_in_whole_datasets / (1 - (1 / CV_N_SPLITS))
  param_search.conduct_xgb_cv(grid_params, tree_method, valid_size_in_trainval)
