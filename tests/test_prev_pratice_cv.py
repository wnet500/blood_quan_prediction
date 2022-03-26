from sklearn.model_selection import RepeatedKFold

from main.prev_practice_cv import conduct_prev_practice_cv
from tests import __train_data_file__ as train
from tests import __test_data_file__ as test

CV_N_SPLITS = 5
CV_N_REPEATS = 10


def test_prev_practice_cv():
  cv = RepeatedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=0)
  conduct_prev_practice_cv(train, test, cv)
