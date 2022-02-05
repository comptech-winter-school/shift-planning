import pandas as pd
import numpy as np
import holidays
import datetime
import category_encoders as ce
from datetime import date


def get_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['date'])

def get_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # добавляем нулевые заказы по часам
    df = df.pivot(*df).stack(dropna=False).reset_index(name='orders_cnt')
    df = df.fillna(0)

    # суммируем по часам
    df = df.groupby([df.delivery_area_id, df.date.dt.date]).orders_cnt.agg('sum')
    df = df.to_frame('target').reset_index()
    
    # удаляем дни с нулевым количеством заказов
    df = df.loc[df.target != 0]

    # df['week_sum'] = df.iloc[:, 2:9].sum(axis=1)
    # df['week_min'] = df.iloc[:, 2:9].min(axis=1)
    # df['week_max'] = df.iloc[:, 2:9].max(axis=1)
    # df['week_mean'] = df.iloc[:, 2:9].mean(axis=1)
    # df['week_std'] = df.iloc[:, 2:9].std(axis=1)

    # features связанные со временем
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
#     df['month'] = df['date'].dt.month
#     df['week'] = df['date'].dt.isocalendar().week
    df['weekday_name'] = df['date'].dt.day_name()
#     df['month_name'] = df.date.dt.month_name()
   
    cat_columns = [col for col in df.columns if (df[col].dtype == "O")]
    encoder = ce.OneHotEncoder(cols=cat_columns).fit(df)
    df = encoder.transform(df)

#     df.drop(['weekday'], axis=1, inplace=True)
    
    return df

def get_prev_future_features(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(6, -1, -1):
        df[f'prev_{i + 1}'] = df.groupby('delivery_area_id')['target'].shift(periods=i)
    for i in range(1, 8):
        df[f'future_{i}'] = df.groupby('delivery_area_id')['target'].shift(periods=-i)
    df.dropna(subset=[f'prev_{i}' for i in range(1, 8)], inplace=True)
    df.drop(['target'], axis=1, inplace=True)
    
    df['week_median'] = df[[f'prev_{i}' for i in range(1, 8)]].median(axis=1)
    
    features = [f'prev_{i}' for i in range(1, 8)] + [f'future_{i}' for i in range(1, 8)]
    df[features] = df[features].div(df['week_median'].values, axis=0)
    
    return df

def get_split_and_features(df: pd.DataFrame, validation=False) -> tuple:
    today = df.date.dt.date.max()
    df = get_features(df)
    
    if validation:  
        
        df_train = df.loc[(today - df.date.dt.date).dt.days >= 14]
        df_train = get_prev_future_features(df_train)
        
        df_val = df.loc[(today - df.date.dt.date).dt.days < 21]
        df_val = get_prev_future_features(df_val)
        
        y_val = df_val[['delivery_area_id', 'date'] + [f'future_{i}' for i in range(1, 8)]]
        X_val = df_val.drop([f'future_{i}' for i in range(1, 8)], axis=1)
        
    else:
        
        df_train = df.copy()
        df_train = get_prev_future_features(df_train)
        
        df_test = df.loc[(today - df.date.dt.date).dt.days < 7]
        df_test = get_prev_future_features(df_test)
        
        X_test = df_test.drop([f'future_{i}' for i in range(1, 8)], axis=1)

    y_train = df_train[['delivery_area_id', 'date'] + [f'future_{i}' for i in range(1, 8)]]
    X_train = df_train.drop([f'future_{i}' for i in range(1, 8)], axis=1)

    return (X_train, X_test, y_train) if not validation else (X_train, X_val, y_train, y_val)

def get_hours_distribution(df_pred_week: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    date = df_pred_week.date.dt.date.values.tolist()[0] + datetime.timedelta(days=1)
    orders = orders.pivot(*orders).stack(dropna=False).reset_index(name='orders_cnt')
    orders = orders.fillna(0)
    
    orders_prev_week = orders.loc[
        ((date - orders.date.dt.date).dt.days <= 7) & ((date - orders.date.dt.date).dt.days > 0)
    ]
    
    orders_prev_week = orders_prev_week.loc[
        orders_prev_week.delivery_area_id.isin(df_pred_week.delivery_area_id.values)
    ]
    
    orders_prev_week['%'] = orders_prev_week.orders_cnt / orders_prev_week.groupby(
        [orders_prev_week.delivery_area_id, orders_prev_week.date.dt.date]
    ) \
    .orders_cnt \
    .transform('sum')
    
    orders_prev_week['%'] = orders_prev_week['%'].fillna(0)
    
    df_pred_hours = orders_prev_week.copy().reset_index(drop=True)
    df_pred_hours.date = df_pred_hours.date + datetime.timedelta(days=7)
    
    counts = df_pred_hours.groupby([df_pred_hours.delivery_area_id, df_pred_hours.date.dt.date]).size() \
    .to_frame('counts') \
    .reset_index()['counts'] \
    .values
    
    df_pred_hours['target'] = pd.Series(
        np.repeat(df_pred_week[[f'future_{i + 1}' for i in range(7)]].values.flatten(), repeats=counts)
    )
    
    df_pred_hours.orders_cnt = df_pred_hours['%'].values * df_pred_hours['target'].round()
    df_pred_hours.orders_cnt = df_pred_hours.orders_cnt.round()
    df_pred_hours.drop(['%', 'target'], axis=1, inplace=True)
    
    return df_pred_hours
    
def get_unscaled_data(df: pd.DataFrame, statistic: pd.Series) -> pd.DataFrame:
    df[[f'future_{i + 1}' for i in range(7)]] = df[[f'future_{i + 1}' for i in range(7)]].multiply(statistic.values, axis=0)
    return df

def get_hours_score(df_pred: pd.DataFrame, real_distrib: pd.DataFrame) -> np.float64:
    dates = df_pred.date.dt.date.unique().tolist()
    scores = []
    for date in dates[:-1]:
        hours_distribution = get_hours_distribution(df_pred[df_pred.date.dt.date == date], real_distrib)
        count_score = pd.merge(
            hours_distribution,
            real_distrib,
            left_on=['delivery_area_id', 'date'],
            right_on=['delivery_area_id', 'date'],
            how='left'
        )
        count_score = count_score[~count_score['orders_cnt_y'].isna()]
        print(count_score)
        y_pred = count_score['orders_cnt_x'].values
        y_true = count_score['orders_cnt_y'].values
        scores.append(np.mean(np.abs((y_true - y_pred) / y_true)))
        print(scores[-1])
        
    return np.mean(scores)
    