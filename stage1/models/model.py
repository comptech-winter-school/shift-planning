import numpy as np
import pandas as pd
import datetime
import holidays
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from datetime import date

class S1Regressor(BaseEstimator):
    def __init__(self, modelType: str):
        self.models = []
        for i in range(7): 
            if (modelType == 'LR'): self.models.append(LinearRegression())
            if(modelType == 'LGBM'): self.models.append(LGBMRegressor(
                objective='MAPE',   
                n_estimators=150, 
                num_leaves=35, 
                learning_rate=0.3
                ))
            else: self.models.append(LinearRegression())
                
    def __add_features(self, df: pd.DataFrame, delta: int) -> pd.DataFrame:
        ru_holidays = holidays.Russia()
        def check_holiday(date: date) -> int:
            return date in ru_holidays

        def days_from_last_holiday(date: date) -> int:
            temp = []
            for day in ru_holidays:
                temp.append((day-date).days)

            return max(x for x in temp if x < 0)

        def days_to_next_holiday(date: date) -> int:
            temp = []
            for day in ru_holidays:
                temp.append((day-date).days)
            if(all(i<0 for i in temp)):
                return min(365+x for x in temp)
            else:
                return min(x for x in temp if x >= 0)
        
        df['curr_date'] = (df['date'] + datetime.timedelta(days=delta)).dt.date
        df['is_holiday'] = df.curr_date.apply(check_holiday).astype(int)
        df['last_holiday'] = df['curr_date'].apply(days_from_last_holiday)
        df['next_holiday'] = df['curr_date'].apply(days_to_next_holiday)
        df['last_holiday_weekday'] = (pd.to_datetime(df['curr_date']) + pd.to_timedelta(df['last_holiday'], 'd')).dt.weekday
        df['next_holiday_weekday'] = (pd.to_datetime(df['curr_date']) + pd.to_timedelta(df['next_holiday'], 'd')).dt.weekday
        df.drop(['curr_date'], axis=1, inplace=True)
        return df
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        for i in range(7):
            X_train_fixed = X_train.loc[y_train[f'future_{i + 1}'].notnull()]
            y_train_fixed = y_train.loc[y_train[f'future_{i + 1}'].notnull()]
            X_train_fixed = self.__add_features(X_train_fixed, i+1)
            
            X_train_fixed = X_train_fixed.drop(['delivery_area_id', 'date'], axis=1)
            y_train_fixed = y_train_fixed.drop(['delivery_area_id', 'date'], axis=1)
            
            self.models[i].fit(X_train_fixed, y_train_fixed[f'future_{i + 1}'])
        
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X_test_fixed = X_test.drop(['delivery_area_id', 'date', 'weekday'], axis=1)
        y_preds = []
        for i in range(7):
            y_pred = self.models[i].predict(self.__add_features(X_test, i+1).drop(['delivery_area_id', 'date'], axis=1))
            y_preds.append(y_pred)
        
        df_pred = pd.DataFrame({f'future_{i + 1}': y_preds[i] for i in range(7)})
        df_pred = df_pred.set_index(X_test.index)
        df_pred[['delivery_area_id', 'date']] = X_test[['delivery_area_id', 'date']]
        df_pred = df_pred[df_pred.columns.tolist()[-2:] + df_pred.columns.tolist()[:-2]]
        
        return df_pred
    
    def score(self, y_test: pd.DataFrame, y_pred: list) -> np.float64:
        scores = []
        for i in range(7):
            scores.append(np.mean(np.abs((y_test[f'future_{i + 1}'] -  y_pred[f'future_{i + 1}']) / y_test[f'future_{i + 1}'])))            
        return np.mean(scores)
