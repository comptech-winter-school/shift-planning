import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stage3.models.solver import get_solution
from stage3.models.pipeline import get_mip_coefficients

def get_partners_distribution_day(pred_partners: pd.DataFrame, date: datetime.date, area_id: int):
    df_temp = pred_partners[(pred_partners.date.dt.date == date) & (pred_partners.delivery_area_id == area_id)]
    c, A, h = get_mip_coefficients(df_temp)
    x_vars = get_solution(c, A, h)
    num_of_hours = df_temp.date.dt.hour.max() - df_temp.date.dt.hour.min() + 1
    sum_work_hours = A.T @ x_vars
    plt.hist([list(range(0, num_of_hours,1))]*2, weights=[h, sum_work_hours])

def get_partners_distribution(pred_partners: pd.DataFrame):
    dates = pred_partners.date.dt.date.unique()
    areas = pred_partners.delivery_area_id.unique()
    
    res = []
    
    for area in areas:
        for date in dates:
            df_temp = pred_partners[(pred_partners.date.dt.date == date) & (pred_partners.delivery_area_id == area)]
            c, A, h = get_mip_coefficients(df_temp)
            x_vars = get_solution(c, A, h)
            min_hour = df_temp.date.dt.hour.min()
            max_hour = df_temp.date.dt.hour.max()
            num_of_hours = max_hour - min_hour + 1
            sum_work_hours = A.T @ x_vars
            for i in range(len(x_vars)):
                if x_vars[i] != 0:
                    min_index = list(A[i, :]).index(1)
                    max_index = num_of_hours - list(A[i, :])[::-1].index(1)
                    start_time = min_hour + min_index
                    end_time = min_hour + max_index
                    for i in range(int(round(x_vars[i]))):
                        res.append([
                            area, 
                            datetime.datetime(date.year, date.month, date.day, start_time), 
                            datetime.datetime(date.year, date.month, date.day, end_time)
                            ])
    return pd.DataFrame(res, columns=['delivery_area_id', 'shift_start_date', 'shift_end_date'])
