import pandas as pd

from . import consts


split_date = consts.SPLIT_DATE

def split_data(df):
    train = df[df['Date'] <= split_date].copy()
    test = df[df['Date'] > split_date].copy()

    page_stats = train.groupby('Page')['Visits'].agg(['median', 'std']).reset_index()
    page_stats.columns = ['Page', 'page_median', 'page_std']

    train = train.merge(page_stats, on='Page', how='left')
    test = test.merge(page_stats, on='Page', how='left')

    daily_mean = train.groupby('Date')['Visits'].mean()

    train['global_mean'] = train['Date'].map(daily_mean.shift(1))
    test['global_mean'] = test['Date'].map(daily_mean.shift(1))

    page_mean = train.groupby('Page')['Visits'].mean()

    bins = pd.qcut(page_mean, q=5, labels=['very_low','low','medium','high','very_high'])
    page_to_bin = dict(zip(page_mean.index, bins))

    train['popularity_bin'] = train['Page'].map(page_to_bin).astype('category')
    test['popularity_bin'] = test['Page'].map(page_to_bin).astype('category')

    page_std = train.groupby('Page')['Visits'].std()

    bins_std = pd.qcut(page_std, q=3, labels=['stable','medium','unstable'])
    page_to_std_bin = dict(zip(page_std.index, bins_std))

    train['volatility_bin'] = train['Page'].map(page_to_std_bin).astype('category')
    test['volatility_bin'] = test['Page'].map(page_to_std_bin).astype('category')

    for col in ['page_median','page_std','global_mean']:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    features = consts.FEATURES + consts.CAT_FEATURES
    target = consts.TARGET
    
    global_mean = train['Visits'].mean()
    counts = train['language'].value_counts()
    te = (train.groupby('language')['Visits'].mean() * counts + global_mean * 10) / (counts + 10)
    train['lang_te'] = train['language'].map(te)
    test['lang_te'] = test['language'].map(te)

    train = train.loc[:, ~train.columns.duplicated()].copy()
    test = test.loc[:, ~test.columns.duplicated()].copy()
    for col in consts.CAT_FEATURES:
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")
    for col in consts.CAT_FEATURES:
        train[col] = train[col].fillna("unknown").astype("category")
        test[col] = test[col].fillna("unknown").astype("category")

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    X_train['Date'] = train['Date']
    X_train['Page'] = train['Page']
    X_test['Date'] = test['Date']
    X_test['Page'] = test['Page']

    return X_test, X_train, y_test, y_train
