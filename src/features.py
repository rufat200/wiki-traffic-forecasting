import numpy as np
import pandas as pd

from . import consts
import slope

def prepare_features():
    df = pd.read_csv(consts.PATH)
    unique_pages = df['Page'].unique() # получаем список уникальных страниц
    sample_pages = pd.Series(unique_pages).sample(n=20_000, random_state=42) # семплируем именно список имен, берем 20 000 случайных названий
    df_subset = df[df["Page"].isin(sample_pages)].copy() # берем историю для выбранных объектов


    df_long = df_subset.melt(id_vars=['Page'], var_name='Date', value_name='Visits') # превращаем колонки в строки

    df_long['Date'] = pd.to_datetime(df_long['Date'].astype(str).str[:-2], format='%Y%m%d')
    df_long['Visits'] = np.log1p(df_long['Visits'].fillna(0)).astype(np.float32)

    df_long['lag_1'] = df_long.groupby('Page')['Visits'].shift(1)
    df_long['lag_2'] = df_long.groupby('Page')['Visits'].shift(2)
    df_long['lag_7'] = df_long.groupby('Page')['Visits'].shift(7)
    df_long['lag_14'] = df_long.groupby('Page')['Visits'].shift(14)
    df_long['lag_21'] = df_long.groupby('Page')['Visits'].shift(21)

    df_long['day_of_week'] = df_long['Date'].dt.dayofweek
    df_long['day_of_month'] = df_long['Date'].dt.day

    df_long['rolling_mean_7'] = df_long.groupby('Page')['Visits'].transform(lambda x: x.rolling(7).mean().shift(1))
    df_long['rolling_std_7']  = df_long.groupby('Page')['Visits'].transform(lambda x: x.rolling(7).std().shift(1))
    df_long['rolling_max_7']  = df_long.groupby('Page')['Visits'].transform(lambda x: x.rolling(7).max().shift(1))

    df_long['z_7'] = (df_long['Visits'] - df_long['rolling_mean_7']) / (df_long['rolling_std_7'] + 1e-6) # Z-score
    df_long['diff_1_7'] = df_long['lag_1'] - df_long['lag_7']

    df_long['slope_7'] = df_long.groupby('Page')['Visits'].transform(lambda x: slope.rolling_slope(x.values.copy(), 7)) # наклон тренда

    df_long['diff_1'] = df_long['Visits'] - df_long['lag_1']
    df_long['diff_2'] = df_long['lag_1'] - df_long['lag_2']
    df_long['diff_1_2'] = df_long['diff_1'] - df_long['diff_2']

    daily_mean = df_long.groupby('Date')['Visits'].mean()
    df_long['global_mean'] = df_long['Date'].map(daily_mean.shift(1))

    df_model = df_long.dropna().copy()
    df_model['is_weekend'] = df_model['day_of_week'].isin([5, 6]).astype(np.int8)


    def extract_features(df):
        # Разделяем строку с конца
        parts = df['Page'].str.split('_')
        df['agent'] = parts.str[-1].astype('category')
        df['access'] = parts.str[-2].astype('category')
        df['language'] = parts.str[-3].str.split('.').str[0].astype('category')
        return df

    df_model = extract_features(df_model)
    return df_model
