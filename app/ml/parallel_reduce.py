from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ParallelReduceClassifier:
    def __init__(self, estimators: Sequence[Pipeline]) -> None:
        if not estimators:
            raise ValueError("At least one fitted estimator is required.")

        self.estimators_ = list(estimators)
        first_classifier = self.estimators_[0].named_steps["classifier"]
        self.classes_ = np.asarray(first_classifier.classes_)
        self.feature_names_in_ = np.asarray(
            self.estimators_[0].feature_names_in_,
            dtype=object,
        )

        for estimator in self.estimators_[1:]:
            classes = estimator.named_steps["classifier"].classes_
            if not np.array_equal(classes, self.classes_):
                raise ValueError(
                    "All reduced estimators must be trained on the same classes."
                )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing_columns = [
                column
                for column in self.feature_names_in_
                if column not in X.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required features: {missing_columns}"
                )
            X = X.loc[:, self.feature_names_in_]

        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)

        for estimator in self.estimators_:
            probabilities += estimator.predict_proba(X)

        return probabilities / len(self.estimators_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
