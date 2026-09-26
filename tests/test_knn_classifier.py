import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def test_subjects_are_not_shared_between_folds():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 4))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    subjects = np.repeat(np.arange(6), 2)

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    for train_idx, test_idx in splitter.split(X, y, groups=subjects):
        assert set(subjects[train_idx]).isdisjoint(set(subjects[test_idx]))
