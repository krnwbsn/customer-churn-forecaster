import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class TenureBucket(BaseEstimator, TransformerMixin):
    """
    Divide tenure into buckets (0-6, 7-12, 13-24, 25-48, 49-72).
    """
    def __init__(self, bins=[0, 6, 12, 24, 48, 72]):
        # store the edges for tenure buckets
        self.bins = bins

    def fit(self, X, y=None):
        # no fitting needed, return self
        return self

    def transform(self, X):
        # work on a copy to avoid modifying the original DataFrame
        X = X.copy()
        # assign each tenure value to a bucket label
        X['TenureBucket'] = pd.cut(
            X['tenure'],
            bins=self.bins,
            labels=[
                f"{self.bins[i]}-{self.bins[i+1]}"
                for i in range(len(self.bins) - 1)
            ],
            include_lowest=True
        )
        # return the transformed DataFrame
        return X
