import pandas as pd
import numpy as np
import holidays
import datetime
import category_encoders as ce
from datetime import date

def get_data(path_orders: str, path_partners_delays: str) -> pd.DataFrame:
    df_orders = pd.read_csv(path_orders, parse_dates=['date'])
    df_partners_delays = pd.read_csv(path_partners_delays, parse_dates=['dttm'])
    
    return df_orders, df_partners_delays

def get_features(df_orders: pd.DataFrame, df_partners_delays=None, validation=False) -> pd.DataFrame:
    if df_partners_delays is not None:
        df = pd.merge(
            df_orders,
            df_partners_delays,
            left_on=['delivery_area_id', 'date'],
            right_on=['delivery_area_id', 'dttm'],
            how='left'
        )
        df.drop(['dttm'], axis=1, inplace=True)
    else:
        df = df_orders
        
    df['perc'] = df['orders_cnt'] / df['partners_cnt']
    df['_perc'] = df['partners_cnt'] / df['orders_cnt']

    df['hour'] = df.date.dt.hour
    b = [0, 4, 8, 12, 16, 20, 24]
    l = ['Late Night', 'Early Morning','Morning','Noon','Eve','Night']
    df['session'] = pd.cut(df['hour'], bins=b, labels=l, include_lowest=True)

    cat_columns = [col for col in df.columns if (df[col].dtype.name == 'category')]
    encoder = ce.OneHotEncoder(cols=cat_columns).fit(df)
    df = encoder.transform(df)
    
    return df
    
def get_train_test_split(df: pd.DataFrame, validation=False) -> pd.DataFrame:
    last_day = df.date.dt.date.max()
    if validation:
        df_train = df.loc[((last_day - df.date.dt.date).dt.days >= 7)]
        df_test = df.loc[((last_day - df.date.dt.date).dt.days < 7) & (df.delay_rate == 0)]
        
        y_test = df_test[['delivery_area_id', 'date', 'partners_cnt', 'delay_rate']]
        X_test = df_test.drop(['delay_rate'], axis=1)
    else:
        df_train = df

    df_train2 = df_train[df_train.delay_rate > 0]
    df_train2['orders_cnt'] = df_train2['orders_cnt'] - df_train2['orders_cnt'] * df_train2['delay_rate']
    df_train2['delay_rate'] = 0
    df_train = pd.concat([df_train, df_train2])
    
    X_train = df_train.drop(['delay_rate'], axis=1)
    y_train = df_train[['delivery_area_id', 'date', 'delay_rate']]

    return (X_train, y_train) if not validation else (X_train, X_test, y_train, y_test)

def optimal_partners(df: pd.DataFrame, model, validation=False) -> pd.DataFrame:
    min_partners = 1
    max_partners = 50
    df = df.loc[df['orders_cnt'] != 0]
    df_ranged = pd.DataFrame(
        np.repeat(df.values, repeats=(max_partners - min_partners + 1), axis=0),
        columns=df.columns
    )
    
    df_ranged['partners_cnt'] = np.tile(
        np.arange(min_partners, max_partners + 1),
        reps=int(df_ranged.shape[0] / (max_partners - min_partners + 1))
    )
    
    if not validation:
        df_ranged = get_features(df_ranged)
    else:
        df_ranged['perc'] = df_ranged['orders_cnt'] / df_ranged['partners_cnt']
        df_ranged['_perc'] = df_ranged['partners_cnt'] / df_ranged['orders_cnt']

    df_pred = model.predict(df_ranged)

    df_pred = df_pred[df_pred.delay_rate < 0.05].reset_index()
    df_pred = df_pred.groupby(['delivery_area_id', 'date']).first().reset_index()
    
    df_partners = df_ranged.loc[df_pred['index']]
    df_partners = df_partners[['delivery_area_id', 'date', 'partners_cnt']].reset_index(drop=True)
    
    df_partners = df_partners.pivot(*df_partners).stack(dropna=False).reset_index(name='partners_cnt')
    df_partners = df_partners.fillna(0)
    
    return df_partners
        

def get_delay_rate_score(y_test: pd.DataFrame, df_pred: pd.DataFrame, ) -> float:
    score = np.mean(np.abs(df_pred['delay_rate'] - y_test['delay_rate']))
    return f'MAE [delay_rate]: {score}'

def get_partners_score(y_test: pd.DataFrame, df_pred: pd.DataFrame, ) -> float:
    score = np.mean(
        np.abs(df_pred['partners_cnt'].values - y_test['partners_cnt'].values) / y_test['partners_cnt'].values
    )
    return f'MAPE [partners_cnt]: {score}'
