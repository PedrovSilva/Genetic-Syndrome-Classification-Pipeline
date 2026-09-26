import numpy as np

from src.models.knn_classifier import KNNClassifier


def test_subjects_are_not_shared_between_folds():
    X = np.random.default_rng(42).normal(size=(20, 4))
    y = np.repeat([0, 1], 10)
    subjects = np.arange(20)

    classifier = KNNClassifier(X, y, subjects, k_range=range(1, 2), cv_folds=5)

    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in splitter.split(X, y, groups=subjects):
        assert set(subjects[train_idx]).isdisjoint(set(subjects[test_idx]))
