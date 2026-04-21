import numpy as np
import lightgbm as lgb

from . import consts


N_ESTIMATORS = consts.N_ESTIMATORS
LRN = consts.LRN
DATE = consts.DATE
def train_my_model(X_train, y_train, X_test, y_test):
    model = lgb.LGBMRegressor(
        objective='regression',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_estimators=N_ESTIMATORS,
        learning_rate=LRN,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    drop_cols = ['Date', 'Page']
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    for col in consts.CAT_FEATURES:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    weights = np.sqrt(np.expm1(y_train))
    weights = np.clip(weights, 1, 100)

    evals_result = {}
    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=
        [(X_train, y_train), 
            (X_test, y_test)],
        eval_names=['train', 'valid'],
        eval_metric='rmse',
        categorical_feature=consts.CAT_FEATURES,
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(stopping_rounds=50, verbose=True)
        ]
    )
    import joblib
    joblib.dump(model, f'./media/model_weights/lgbm_traffic_model-{N_ESTIMATORS}-{LRN}-{DATE}.pkl')
    return model, evals_result, X_test

