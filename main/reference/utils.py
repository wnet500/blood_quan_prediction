import numpy as np

import scipy.stats as st
from sklearn.metrics import r2_score


def adjust_pred_value(x):
    x = np.where(x < 0, 0, x)
    x = np.where(x < 3.5, np.round(x), np.ceil(x))

    return x


def get_adjusted_r2(true_vals, predicted_vals, num_of_vals):
  adj_r2 = 1 - (1 - r2_score(true_vals, predicted_vals)) * (len(true_vals) - 1) / (len(true_vals) - num_of_vals - 1)

  return adj_r2


def get_95_conf_interval(x):
  lower, upper = st.t.interval(
      alpha=0.95,
      df=len(x) - 1,
      loc=np.mean(x),
      scale=st.sem(x)
  )
  return [np.mean(x), lower, upper]
