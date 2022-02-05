import warnings
import datetime
import pandas as pd
import os
from stage1.main import get_next_week
from stage2.main import get_optimal_partners
from stage3.main import get_partners_distribution
from utils import path_orders, path_partners, path_pred_orders, path_pred_partners

warnings.filterwarnings('ignore')

def get_week() -> (pd.DataFrame, pd.DataFrame):
    if not os.path.exists(path_pred_orders) and not os.path.exists(path_pred_partners): 
        pred_orders = get_next_week(path_orders)
        pred_orders.to_csv(path_pred_orders, index=False)
        pred_partners = get_optimal_partners(path_orders, path_partners, pred_orders)
        pred_partners.to_csv(path_pred_partners, index=False)
    else:
        pred_orders = pd.read_csv(path_pred_orders, parse_dates=['date'])
        pred_partners = pd.read_csv(path_pred_partners, parse_dates=['date'])

    pred_orders_partners = pd.merge(
        pred_orders,
        pred_partners,
        on=['delivery_area_id', 'date']
    )
    
    shifts = get_partners_distribution(pred_partners)
    return pred_orders_partners, shifts
