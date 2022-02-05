from json.tool import main
import pandas as pd
import datetime
from stage1.models.model import S1Regressor
from stage1.models.pipeline import *

def get_next_week(path: str) -> pd.DataFrame:
    df = get_data(path)
    X_train, X_test, y_train = get_split_and_features(df)
    model = S1Regressor('LGBM')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = get_unscaled_data(y_pred, X_test.week_median)
    hours_distribution = get_hours_distribution(y_pred, df)
    
    return hours_distribution
