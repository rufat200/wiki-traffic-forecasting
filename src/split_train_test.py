from . import consts


split_date = consts.SPLIT_DATE

def split_data(df):
    train = df[df['Date'] <= split_date]
    test = df[df['Date'] > split_date]

    page_stats = train.groupby('Page')['Visits'].agg(['median', 'std']).reset_index()
    page_stats.columns = ['Page', 'page_median', 'page_std']

    train = train.merge(page_stats, on='Page', how='left')
    test = test.merge(page_stats, on='Page', how='left')

    train['page_median'] = train['page_median'].fillna(0)
    train['page_std'] = train['page_std'].fillna(0)

    test['page_median'] = test['page_median'].fillna(0)
    test['page_std'] = test['page_std'].fillna(0)

    features = consts.FEATURES
    target = consts.TARGET

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    return X_test, X_train, y_test, y_train
