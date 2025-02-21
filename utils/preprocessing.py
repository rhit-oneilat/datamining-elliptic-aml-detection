"""Preprocessing utilities including custom transformers"""

from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline


class EnhancedSampler(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy
        self.pipeline = ImbPipeline(
            [
                ("adasyn", ADASYN(sampling_strategy=sampling_strategy)),
                ("tomek", TomekLinks()),
            ]
        )

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is not None:
            X_resampled, y_resampled = self.pipeline.fit_resample(X, y)
            return X_resampled, y_resampled
        return X
