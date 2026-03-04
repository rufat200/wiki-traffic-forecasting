import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
)

import lightgbm as lgb
from src.consts import (
    N_ESTIMATORS,
    LRN,
    DATE,
)
from src.fit_model_lightgbm import train_my_model
from src.split_train_test import split_data
from src.features import prepare_features


def main():

    df_model = prepare_features()
    X_test, X_train, y_test, y_train = split_data(df_model)

    model, evals_result = train_my_model(X_train, y_train, X_test, y_test)
    # import joblib
    # model = joblib.load('./media/model_weights/lgbm_traffic_model-900-0.09-2026-03-04_15-44-25.pkl')
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log) # Прогноз в реальных числах
    y_test_real = np.expm1(y_test)  


    def mean_absolute_scaled_error(y_true, mae):
        naive_mae = np.mean(np.abs(np.diff(y_true))) + 1e-10
        return mae/naive_mae
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    print(f"Симметричная средняя абсолютная процентная ошибка SMAPE: {symmetric_mean_absolute_percentage_error(y_test_real, preds):.5f}%")
    mae = mean_absolute_error(y_test_real, preds)
    mse = mean_squared_error(y_test_real, preds)
    rmse = root_mean_squared_error(y_test_real, preds)
    r2 = r2_score(y_test_real, preds)*100
    mase = mean_absolute_scaled_error(y_test_real, mae)

    print(f"Средняя абсолютная ошибка (MAE): {mae:.5f}")
    print(f"Средняя квадратичная ошибка (MSE): {mse:.5f}")
    print(f"Корневая средняя квадртичная ошибка (RMSE): {rmse:.5f}")
    print(f"Коэффициент Детерминации (R^2): {r2:.5f}%")
    print(f"Средняя абсолютная скалярная ошибка (MASE): {mase:.5f}")


    lgb.plot_importance(model, importance_type='gain', figsize=(10, 6), title='Важность признаков (Feature Importance)')
    plt.tight_layout()
    plt.savefig(f"./media/graphs/model_importances-{N_ESTIMATORS}-{LRN}-{DATE}.png")

    plt.figure(figsize=(10, 6))
    epochs = len(evals_result['valid']['rmse'])
    x_axis = range(0, epochs)
    plt.plot(x_axis, evals_result['train']['rmse'], label='Обучение (Train)')
    plt.plot(x_axis, evals_result['valid']['rmse'], label='Валидация (Validation)')
    plt.title('Learning Curve (RMSE, log scale target)')
    plt.xlabel('Количество деревьев (Итерации)')
    plt.ylabel('RMSE (log1p(Visits))')
    plt.legend()
    plt.grid(True, alpha=0.3)
    best_iter = model.best_iteration_
    plt.axvline(x=best_iter, color='r', linestyle='--', label='Лучшая итерация')
    plt.savefig(f"./media/graphs/learning_curve-{N_ESTIMATORS}-{LRN}-{DATE}.png") 

    plt.figure(figsize=(12, 6))
    plt.plot(np.expm1(y_test.values[:100]), label='Реальные (Actual)', alpha=0.7, linewidth=2)
    plt.plot(preds[:100], label='Прогноз (Predicted)', alpha=0.7, linestyle='--')
    plt.legend()
    plt.title(f"Сравнение реальных данных и прогноза (MAE: {mae:.2f})")
    plt.ylabel("Количество просмотров")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./media/graphs/real_vs_pred-{N_ESTIMATORS}-{LRN}-{DATE}.png")

    # График распределения ошибок
    residuals = y_test_real - preds
    print("Min:", residuals.min())
    print("Max:", residuals.max())
    print("Std:", residuals.std())
    print("95 percentile:", np.percentile(residuals, 95))
    print("99 percentile:", np.percentile(residuals, 99))
    print(np.percentile(residuals, [1, 99]))
    low, high = np.percentile(residuals, [1, 99])
    filtered = residuals[(residuals >= low) & (residuals <= high)]

    plt.figure(figsize=(10, 6))
    sns.histplot(filtered, bins=40, color='skyblue', edgecolor='black')
    # plt.yscale('log')
    plt.title("Residuals (1–99 percentile)")
    plt.ylabel("Count")
    plt.xlabel('Ошибка (Реальность - Прогноз)')
    plt.xlim(-500, 500)
    plt.grid(alpha=0.3)
    plt.savefig(f"./media/graphs/residuals_hist_improved-{N_ESTIMATORS}-{LRN}-{DATE}.png")

if __name__ == '__main__':
    main()
