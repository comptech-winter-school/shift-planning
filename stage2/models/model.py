import numpy as np
import pandas as pd
import datetime
import holidays
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from datetime import date

class S2Regressor(BaseEstimator):
    def __init__(self, model_type: str):
        if model_type == 'LR':
            self.model = LinearRegression(normalize=False)
        if model_type == 'LGBM':
            self.model = LGBMRegressor
            
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        X_train_fixed = X_train.drop(['delivery_area_id', 'date', 'hour'], axis=1)
        y_train_fixed = y_train.drop(['delivery_area_id', 'date'], axis=1)
        self.model.fit(X_train_fixed, y_train_fixed)
        
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X_test_fixed = X_test.drop(['delivery_area_id', 'date', 'hour'], axis=1)
        y_pred = self.model.predict(X_test_fixed)
        y_pred[y_pred < 0] = 0
        
        df_pred = pd.DataFrame({'delay_rate': y_pred.reshape(-1)}, index=X_test.index)
        df_pred = pd.concat([X_test[['delivery_area_id', 'date']], df_pred], axis=1).reset_index(drop=True)
        
        return df_pred
