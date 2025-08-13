from sklearn.metrics import roc_auc_score
import pandas as pd

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict_proba(X_train)[:, 1]
        return roc_auc_score(y_train, y_pred)
    
    def save_test_predictions(self, X_test, ids, filename):
        pd.DataFrame({
            'SK_ID_CURR': ids,
            'TARGET': self.model.predict_proba(X_test)[:, 1]
        }).to_csv(filename, index=False)

