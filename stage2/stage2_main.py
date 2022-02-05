import pandas as pd
import datetime
from .models.stage2_model import S2Regressor
from .models.stage2_pipeline import *


def get_optimal_partners(path_orders: str, path_partners_delays: str, pred_orders: pd.DataFrame) -> pd.DataFrame:
    df_orders, df_partners_delays = get_data(path_orders, path_partners_delays)
    df = get_features(df_orders, df_partners_delays=df_partners_delays)
    X_train, y_train = get_train_test_split(df)
    
    model = S2Regressor('LR')
    model.fit(X_train, y_train)
    
    partners_pred = optimal_partners(pred_orders, model)
    
    return partners_pred
