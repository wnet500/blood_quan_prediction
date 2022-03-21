import numpy as np
import pandas as pd
import shutil
import torch
import re

from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, ParameterGrid, train_test_split
from torch.utils.data import DataLoader

from main.reference.modeling import ModelTrainer
from main.reference.prepocessing import DataProcessor
from main.reference.torch_modeling_reference import CustomDataset
from main.reference.utils import (
    adjust_pred_value,
    get_adjusted_r2,
    get_95_conf_interval,
    disable_logging_and_userwaring
)


class ParamGridSearch:
  def __init__(
      self,
      train_data_file: str,
      test_data_file: str,
      cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
  ) -> None:
    self.cv = cv
    self.X_trainval, self.X_test, self.y_trainval, self.y_test = DataProcessor(
        train_data_file, test_data_file).make_X_y_datasets()
    self.ouput_dir = Path(__file__).parent.parent.joinpath("output")

  def conduct_lr_cv(self):
    mse_evals = []
    adj_r2_evals = []

    for train_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
      X_train_in, X_test_in = self.X_trainval[train_index], self.X_trainval[test_index]
      y_train_in, y_test_in = self.y_trainval[train_index], self.y_trainval[test_index]

      lr_model = ModelTrainer(X_train_in, y_train_in).train_linear_regression()
      y_pred = adjust_pred_value(lr_model.predict(X_test_in))
      y_pred = np.where(y_pred > 15, 15, y_pred)

      mse = mean_squared_error(y_test_in, y_pred)
      mse_evals.append(mse)

      adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
      adj_r2_evals.append(adj_r2)

    print()
    print("---> Linear Regression MSE(mean of cv): {:.3f})".format(np.mean(mse_evals)))
    print("---> Linear Regression Adj r2(mean of cv): {:.3f})".format(np.mean(adj_r2_evals)))

    lr_mse_mean, lr_mse_95_ci_lower, lr_mse_95_ci_upper = get_95_conf_interval(mse_evals)
    lr_adj_r2_mean, lr_adj_r2_95_ci_lower, lr_adj_r2_95_ci_upper = get_95_conf_interval(adj_r2_evals)

    result_df = pd.DataFrame({
        "lr_mse_cv_results": [mse_evals],
        "lr_adj_r2_results": [adj_r2_evals],
        "lr_mse_mean": [lr_mse_mean],
        "lr_mse_95_ci_lower": [lr_mse_95_ci_lower],
        "lr_mse_95_ci_upper": [lr_mse_95_ci_upper],
        "lr_adj_r2_mean": [lr_adj_r2_mean],
        "lr_adj_r2_95_ci_lower": [lr_adj_r2_95_ci_lower],
        "lr_adj_r2_95_ci_upper": [lr_adj_r2_95_ci_upper]
    })

    result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "lr_results.csv")), index=False)

  def conduct_rf_cv(self, grid_params: dict):
    start_time = datetime.now()
    gridsearch_results = []
    print(f"[{start_time}] Start parameter search cv for model...")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []

      for train_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
        X_train_in, X_test_in = self.X_trainval[train_index], self.X_trainval[test_index]
        y_train_in, y_test_in = self.y_trainval[train_index], self.y_trainval[test_index]

        rf_model = ModelTrainer(X_train_in, y_train_in).train_random_forest(param)
        y_pred = adjust_pred_value(rf_model.predict(X_test_in))

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

      gridsearch_results.append((param, mse_evals, adj_r2_evals))

      print()
      print('[{}] param:\n{}'.format(param_ind + 1, param))
      print('---> RandomForest MSE(mean of cv): {:.3f}'.format(np.mean(mse_evals)))
      print('---> Cumulative time: {:.3f} minutes'.format((datetime.now() - start_time).seconds / 60))

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=["param", "rf_mse_cv_results", "rf_adj_r2_cv_results"]
    )
    conf_info_1 = gridsearch_result_df["rf_mse_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_1.columns = ["rf_mse_mean", "rf_mse_95_ci_lower", "rf_mse_95_ci_upper"]
    conf_info_2 = gridsearch_result_df["rf_adj_r2_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_2.columns = ["rf_adj_r2_mean", "rf_adj_r2_95_ci_lower", "rf_adj_r2_95_ci_upper"]

    gridsearch_result_df = pd.concat(
        [gridsearch_result_df, conf_info_1, conf_info_2],
        axis=1
    ).sort_values(by="rf_mse_mean")

    gridsearch_result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "rf_results.csv")), index=False)

  def conduct_xgb_cv(
      self,
      grid_params: dict,
      tree_method: str = "gpu_hist",
      valid_size_in_trainval=1 / 8
  ):
    start_time = datetime.now()
    gridsearch_results = []
    print(f"[{start_time}] Start parameter search cv for model...")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []
      best_ntree_limits = []

      for trainval_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
        X_trainval_in, X_test_in = self.X_trainval[trainval_index], self.X_trainval[test_index]
        y_trainval_in, y_test_in = self.y_trainval[trainval_index], self.y_trainval[test_index]

        X_train_in, X_valid_in, y_train_in, y_valid_in = train_test_split(
            X_trainval_in, y_trainval_in,
            test_size=valid_size_in_trainval,
            random_state=0
        )

        xgb_model = ModelTrainer(X_train_in, y_train_in).train_xgboost(
            param=param,
            tree_method=tree_method,
            eval_set=[(X_valid_in, y_valid_in)],
            n_estimators=10000
        )
        y_pred = adjust_pred_value(xgb_model.predict(X_test_in))

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

        best_ntree_limits.append(xgb_model.best_ntree_limit)

      gridsearch_results.append((param, mse_evals, adj_r2_evals, best_ntree_limits))

      print()
      print('[{}] param:\n{}'.format(param_ind + 1, param))
      print('---> XGB MSE(mean of cv): {:.3f}'.format(np.mean(mse_evals)))
      print('---> Cumulative time: {:.3f} minutes'.format((datetime.now() - start_time).seconds / 60))

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=["param", "xgb_mse_cv_results", "xgb_adj_r2_cv_results", "early_stopping_rounds"]
    )
    gridsearch_result_df["early_stopping_round_mean"] = \
        round(gridsearch_result_df["early_stopping_rounds"].apply(lambda x: np.mean(x)))

    conf_info_1 = gridsearch_result_df["xgb_mse_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_1.columns = ["xgb_mse_mean", "xgb_mse_95_ci_lower", "xgb_mse_95_ci_upper"]
    conf_info_2 = gridsearch_result_df["xgb_adj_r2_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_2.columns = ["xgb_adj_r2_mean", "xgb_adj_r2_95_ci_lower", "xgb_adj_r2_95_ci_upper"]

    gridsearch_result_df = pd.concat(
        [gridsearch_result_df, conf_info_1, conf_info_2],
        axis=1
    ).sort_values(by="xgb_mse_mean")

    gridsearch_result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "xgb_results.csv")), index=False)

  def conduct_ann_cv(
      self,
      grid_params: dict,
      valid_size_in_trainval: float = 1 / 8,
      is_disable_logging_and_userwaring: bool = True
  ):
    if is_disable_logging_and_userwaring:
      disable_logging_and_userwaring()

    shutil.rmtree(str(self.ouput_dir.joinpath("tb_logs")), ignore_errors=True)

    start_time = datetime.now()
    gridsearch_results = []
    print(f"[{start_time}] Start parameter search cv for model...")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []
      best_epochs = []

      for cv_num, (trainval_index, test_index) in enumerate(self.cv.split(self.X_trainval, self.y_trainval)):
        X_trainval_in, X_test_in = self.X_trainval[trainval_index], self.X_trainval[test_index]
        y_trainval_in, y_test_in = self.y_trainval[trainval_index], self.y_trainval[test_index]

        X_train_in, X_valid_in, y_train_in, y_valid_in = train_test_split(
            X_trainval_in, y_trainval_in,
            test_size=valid_size_in_trainval,
            random_state=0
        )

        eval_dataset = CustomDataset(X_valid_in, y_valid_in)
        test_dataset = CustomDataset(X_test_in, y_test_in)

        evalset_loader = DataLoader(eval_dataset, batch_size=2**12, shuffle=False, num_workers=2)
        testset_loader = DataLoader(test_dataset, batch_size=2**13, shuffle=False, num_workers=2)

        logger = TensorBoardLogger(
            save_dir=str(self.ouput_dir.joinpath("tb_logs")),
            name=f"grid_param_{param_ind}",
            version=f"cross_validation_{cv_num}"
        )
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(self.ouput_dir.joinpath(
                f"tb_logs/grid_param_{param_ind}/cross_validation_{cv_num}/checkpoints")),
            filename="ann_{epoch:03d}_{val_loss:.3f}",
            mode="min"
        )

        model_trainer = ModelTrainer(X_train_in, y_train_in)
        trainer, ann_model = model_trainer.train_ann(
            param=param,
            logger=logger,
            checkpoint_cb=checkpoint_cb,
            evalset_loader=evalset_loader,
            has_bar_callback=False
        )

        results = trainer.predict(ann_model, dataloaders=testset_loader)
        y_pred = torch.vstack([results[i][0] for i in range(len(results))]).cpu().numpy()[:, 0]
        y_pred = adjust_pred_value(y_pred)

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

        best_epoch = int(re.search(r'epoch=(?P<epoch>\d*)', checkpoint_cb.best_model_path).group('epoch'))
        best_epochs.append(best_epoch)

      gridsearch_results.append((param, mse_evals, adj_r2_evals, best_epochs))

      print()
      print('[{}] param:\n{}'.format(param_ind + 1, param))
      print('---> ANN MSE(mean of cv): {:.3f}'.format(np.mean(mse_evals)))
      print('---> Cumulative time: {:.3f} minutes'.format((datetime.now() - start_time).seconds / 60))

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=["param", "ann_mse_cv_results", "ann_adj_r2_cv_results", "best_epochs"]
    )
    gridsearch_result_df["best_epoch_mean"] = \
        round(gridsearch_result_df["best_epochs"].apply(lambda x: np.mean(x)))

    conf_info_1 = gridsearch_result_df["ann_mse_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_1.columns = ["ann_mse_mean", "ann_mse_95_ci_lower", "ann_mse_95_ci_upper"]
    conf_info_2 = gridsearch_result_df["ann_adj_r2_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_2.columns = ["ann_adj_r2_mean", "ann_adj_r2_95_ci_lower", "ann_adj_r2_95_ci_upper"]

    gridsearch_result_df = pd.concat(
        [gridsearch_result_df, conf_info_1, conf_info_2],
        axis=1
    ).sort_values(by="ann_mse_mean")

    gridsearch_result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "ann_results.csv")), index=True)
