import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from main.reference.prepocessing import DataProcessor
from main.reference.utils import get_95_conf_interval


def conduct_prev_practice_cv(
    train_data_file: str,
    test_data_file: str,
    cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
) -> None:
  prev_practice_trainval, _, y_trainval, _ = DataProcessor(
      train_data_file,
      test_data_file
  ).make_prev_practice_datasets()

  prev_practice_mse_evals = []
  prev_practice_r2_evals = []
  for train_index, test_index in cv.split(prev_practice_trainval, y_trainval):
    _, prev_practice_test_in = prev_practice_trainval[train_index], prev_practice_trainval[test_index]
    _, y_test_in = y_trainval[train_index], y_trainval[test_index]

    prev_practice_mse = mean_squared_error(y_test_in, prev_practice_test_in)
    prev_practice_r2 = r2_score(y_test_in, prev_practice_test_in)

    prev_practice_mse_evals.append(prev_practice_mse)
    prev_practice_r2_evals.append(prev_practice_r2)

  print()
  print("---> Previous practice MSE(mean of cv): {:.3f})".format(np.mean(prev_practice_mse_evals)))
  print("---> Previous practice r2(mean of cv): {:.3f})".format(np.mean(prev_practice_r2_evals)))

  mse_mean, mse_95_ci_lower, mse_95_ci_upper = get_95_conf_interval(prev_practice_mse_evals)
  r2_mean, r2_95_ci_lower, r2_95_ci_upper = get_95_conf_interval(prev_practice_r2_evals)

  result_df = pd.DataFrame({
      "prev_practice_mse_cv_results": [prev_practice_mse_evals],
      "prev_practice_r2_results": [prev_practice_r2_evals],
      "prev_practice_mse_mean": [mse_mean],
      "prev_practice_mse_95_ci_lower": [mse_95_ci_lower],
      "prev_practice_mse_95_ci_upper": [mse_95_ci_upper],
      "prev_practice_r2_mean": [r2_mean],
      "prev_practice_r2_95_ci_lower": [r2_95_ci_lower],
      "prev_practice_r2_95_ci_upper": [r2_95_ci_upper]
  })

  ouput_dir = Path(__file__).parent.parent.joinpath("output")
  result_df.to_csv(str(ouput_dir.joinpath("prev_practice_cv_results.csv")), index=False)
