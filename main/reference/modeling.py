import numpy as np
import xgboost

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class ModelTrainer:
  def __init__(
      self,
      X_train: np.ndarray,
      y_train: np.ndarray
  ) -> None:
    self.X_train = X_train
    self.y_train = y_train

  def train_linear_regression(self):
    lr_model = LinearRegression()
    lr_model.fit(self.X_train, self.y_train)

    return lr_model

  def train_random_forest(self, param: dict):
    rf_model = RandomForestRegressor(random_state=0, n_jobs=-1, **param)
    rf_model.fit(self.X_train, self.y_train)

    return rf_model

  def train_xgboost(
      self,
      param: dict,
      tree_method: str,
      eval_set: list = None,
      n_estimators: int = 10000
  ):
    xgb_model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        random_state=0,
        n_estimators=n_estimators,
        tree_method=tree_method,
        **param
    )

    if eval_set:
      xgb_model.fit(
          self.X_train, self.y_train,
          early_stopping_rounds=1000,
          eval_set=eval_set,
          eval_metric="rmse",
          verbose=False
      )
    else:
      xgb_model.fit(
          self.X_train, self.y_train,
          verbose=False
      )

    return xgb_model
