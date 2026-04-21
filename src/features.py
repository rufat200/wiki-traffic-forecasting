import numpy as np
import pandas as pd

from . import consts
import slope

def prepare_features():
    df = pd.read_csv(consts.PATH)

    unique_pages = df['Page'].unique()
    sample_pages = pd.Series(unique_pages).sample(n=10_000, random_state=42)
    df_subset = df[df["Page"].isin(sample_pages)].copy()

    df_long = df_subset.melt(id_vars=['Page'], var_name='Date', value_name='Visits')

    df_long['Date'] = pd.to_datetime(df_long['Date'].astype(str).str[:-2], format='%Y%m%d')
    df_long['Visits'] = np.log1p(df_long['Visits'].fillna(0)).astype(np.float32)
    df_long = df_long[df_long.groupby('Page')['Visits'].transform('mean') > 0.1]

    for lag in [1, 2, 7, 14, 21, 28, 30]:
        df_long[f'lag_{lag}'] = df_long.groupby('Page')['Visits'].shift(lag)

    df_long['day_of_week'] = df_long['Date'].dt.dayofweek.astype('category')
    df_long['day_of_month'] = df_long['Date'].dt.day
    df_long['is_weekend'] = df_long['day_of_week'].isin([5, 6]).astype(np.int8)

    df_long['rolling_mean_7'] = df_long.groupby('Page')['Visits'].transform(lambda x: x.shift(1).rolling(7).mean())
    df_long['rolling_std_7']  = df_long.groupby('Page')['Visits'].transform(lambda x: x.shift(1).rolling(7).std())
    df_long['rolling_max_7']  = df_long.groupby('Page')['Visits'].transform(lambda x: x.shift(1).rolling(7).max())

    df_long['z_7'] = (df_long['Visits'] - df_long['rolling_mean_7']) / (df_long['rolling_std_7'] + 1e-6)

    df_long['diff_1'] = df_long['Visits'] - df_long['lag_1']
    df_long['diff_2'] = df_long['lag_1'] - df_long['lag_2']
    df_long['diff_1_2'] = df_long['diff_1'] - df_long['diff_2']
    df_long['diff_1_7'] = df_long['lag_1'] - df_long['lag_7']

    df_long['ratio_1_7'] = df_long['lag_1'] / (df_long['lag_7'] + 1e-6)
    df_long['ratio_7_14'] = df_long['lag_7'] / (df_long['lag_14'] + 1e-6)

    df_long['slope_7'] = df_long.groupby('Page')['Visits'].transform(
        lambda x: slope.rolling_slope(x.values.copy(), 7)
    )
    df_long['ewm_7'] = df_long.groupby('Page')['Visits'].transform(
        lambda x: x.shift(1).ewm(span=7).mean()
    )

    df_long['ewm_30'] = df_long.groupby('Page')['Visits'].transform(
        lambda x: x.shift(1).ewm(span=30).mean()
    )

    parts = df_long['Page'].str.split('_')
    project_lang = parts.str[-3]

    df_long['project'] = project_lang.str.split('.').str[-1].astype('category')
    df_long['language'] = project_lang.str.split('.').str[0].astype('category')

    df_long['page_type'] = df_long['Page'].str.extract(r'^([^:]+):')[0]
    df_long['page_type'] = df_long['page_type'].fillna('article').astype('category')

    df_model = df_long.dropna().copy()
    return df_model
