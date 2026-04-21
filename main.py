import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans'

import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error, 
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

import lightgbm as lgb
from src.consts import (
    N_ESTIMATORS,
    LRN,
    DATE,
    TARGET,
    FEATURES,
    SPLIT_DATE,
    CAT_FEATURES
)
from src.fit_model_lightgbm import train_my_model
from src.split_train_test import split_data
from src.features import prepare_features


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    # sMAPE
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    
    # MASE
    naive_mae = np.mean(np.abs(np.diff(y_true))) + 1e-10
    mase = mae / naive_mae
    
    return {
        "MAE": mae, 
        "RMSE": rmse, 
        "MedAE": medae, 
        "R2": r2, 
        "Bias": bias, 
        "sMAPE": smape, 
        "MASE": mase, 
        "MSE": mse, 
    }



def main():
    df_model = prepare_features()
    X_test, X_train, y_test, y_train = split_data(df_model)

    print(type(X_test['language']))
    print(X_test['language'].dtype)
    print(X_test.columns[X_test.columns.duplicated()])
    test_dates = X_test['Date'].copy()
    test_pages = X_test['Page'].copy()

    model, evals_result, X_test = train_my_model(X_train, y_train, X_test, y_test)

    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_test_real = np.expm1(y_test)

    residuals = y_test_real - preds

    metrics = calculate_metrics(y_test_real, preds)

    print("\n--- Итоговые метрики (REAL SPACE) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}")


    plt.figure(figsize=(10, 6))
    plt.plot(evals_result['train']['rmse'], label="train")
    plt.plot(evals_result['valid']['rmse'], label="valid")
    best_iter = model.best_iteration_
    plt.axvline(best_iter, color='red', linestyle='--')
    plt.title("Learning Curve (RMSE log-space)")
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./media/graphs/learning_curve-{N_ESTIMATORS}-{LRN}-{DATE}.png") 


    plt.figure(figsize=(10, 8))
    ax = lgb.plot_importance(model, importance_type='gain', max_num_features=20)
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(f"./media/graphs/feature_importance-{N_ESTIMATORS}-{LRN}-{DATE}.png")


    plt.figure(figsize=(8, 8))
    sample_idx = np.random.choice(len(y_test_real), 10_000, replace=False)
    x = y_test_real.iloc[sample_idx]
    y = preds[sample_idx]
    sns.scatterplot(x=x, y=y, alpha=0.2, color="black")
    max_val = np.percentile(np.r_[x, y], 99)
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Actual vs Predicted (log-log scale)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"./media/graphs/scatter_fact_pred-{N_ESTIMATORS}-{LRN}-{DATE}.png")


    mask = preds > 1
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=preds[mask], y=np.abs(residuals[mask]), alpha=0.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Error vs Prediction (calibration check, non-zero)")
    plt.xlabel("Predicted")
    plt.ylabel("Absolute Error")
    plt.tight_layout()
    plt.savefig(f"./media/graphs/error_vs_pred-{N_ESTIMATORS}-{LRN}-{DATE}.png")


    analysis_df = X_test.copy()
    analysis_df["error"] = np.abs(residuals)
    bin_order = ['very_low', 'low', 'medium', 'high', 'very_high']
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=analysis_df, x="popularity_bin", y="error", order=bin_order)
    plt.yscale("log")
    plt.title("Error by Popularity Bin")
    plt.tight_layout()
    plt.savefig(f"./media/graphs/error_by_popularity-{N_ESTIMATORS}-{LRN}-{DATE}.png")


    plt.figure(figsize=(12, 6))
    plt.plot(np.expm1(y_test.values[:100]), label='Реальные (Actual)', alpha=0.7, linewidth=2)
    plt.plot(preds[:100], label='Прогноз (Predicted)', alpha=0.7, linestyle='--')
    plt.legend()
    plt.title("Сравнение реальных данных и прогноза")
    plt.ylabel("Количество просмотров")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./media/graphs/real_vs_pred-{N_ESTIMATORS}-{LRN}-{DATE}.png")


    sample_page_acf = test_pages.value_counts().index[0]
    mask_acf = (test_pages == sample_page_acf).values
    page_residuals = y_test.values[mask_acf] - preds_log[mask_acf]
    plt.figure(figsize=(10, 5))
    plot_acf(page_residuals, lags=30)
    plt.title(f"ACF Residuals (log space, page: {sample_page_acf[:40]}...)")
    plt.tight_layout()
    plt.savefig(f"./media/graphs/acf-{N_ESTIMATORS}-{LRN}-{DATE}.png")
    

    residuals = y_test_real - preds
    p1, p99 = np.percentile(residuals, [1, 99])
    residuals_clipped = residuals[(residuals >= p1) & (residuals <= p99)]
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals_clipped, bins=100, kde=True)
    plt.axvline(0, color='red')
    plt.title("Residuals Distribution (real space, 1-99 percentile)")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"./media/graphs/residuals_dist-{N_ESTIMATORS}-{LRN}-{DATE}.png")


    sample_pages = np.random.choice(test_pages.unique(), 3, replace=False)
    plt.figure(figsize=(15, 10))
    for i, page in enumerate(sample_pages):
        mask = test_pages == page
        page_dates = test_dates[mask]
        X_page = X_test[mask.values]  # X_test уже без Date/Page
        preds_page = np.expm1(model.predict(X_page))
        plt.subplot(3, 1, i+1)
        plt.plot(page_dates.values, np.expm1(y_test[mask].values), label='actual')
        plt.plot(page_dates.values, preds_page, '--', label='pred')
        plt.title(page)
        plt.legend()
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./media/graphs/page_forecasts-{N_ESTIMATORS}-{LRN}-{DATE}.png")


if __name__ == '__main__':
    main()
