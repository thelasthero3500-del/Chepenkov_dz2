import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, custom_steps=None):
        self.custom_steps = custom_steps or []
    
    def transform(self, X):
        X = X.copy()
        X = self._basic_clean(X)  
        
        for step in self.custom_steps:
            X = step(X) 
        return X
    
    def _basic_clean(self, X):
        X = X.drop_duplicates(keep='first')
        
        num_cols = X.select_dtypes(include='number').columns
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        return X  