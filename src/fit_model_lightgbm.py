from . import consts
import lightgbm as lgb

N_ESTIMATORS = consts.N_ESTIMATORS
LRN = consts.LRN
DATE = consts.DATE
def train_my_model(X_train, y_train, X_test, y_test):
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=N_ESTIMATORS,
        learning_rate=LRN,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
    )
    evals_result = {}
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
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
    return model, evals_result

