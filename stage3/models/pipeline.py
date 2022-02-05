import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_mip_coefficients(df: pd.DataFrame):
    num_of_hours = df.date.dt.hour.max() - df.date.dt.hour.min() + 1
    work_intervals = list(range(4,9))
    num_of_intervals = sum([num_of_hours - x + 1 for x in work_intervals])
    A = np.zeros((num_of_intervals, num_of_hours))
    A.shape
    row_counter = 0
    for interval in work_intervals:
        new_row = np.zeros(num_of_hours)
        new_row[:interval] = 1
        for i in range(num_of_hours - interval + 1):
            A[row_counter, :] = new_row
            new_row = np.roll(new_row, 1)
            row_counter += 1

    h = df.partners_cnt.to_numpy()
    c = A @ np.ones(num_of_hours)
    
    return c, A, h
