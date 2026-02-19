import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import joblib


path = r"C:\Users\Rufat\.cache\kagglehub\datasets\sandeshbhat\wikipedia-web-traffic-201819\versions\1\Wiki_Page_views.csv"

import pandas as pd

df = pd.read_csv(path)
df = df.sample(20_000, random_state=42)


df_long = df.melt(id_vars=['Page'], var_name='Date', value_name='Visits') # превращаем колонки в строки
import numpy as np


df_long['Date'] = pd.to_datetime(df_long['Date'].astype(str).str[:-2], format='%Y%m%d') # отрезал два нуля (часы) и конвертируем
df_long['Visits'] = np.log1p(df_long['Visits'].fillna(0)).astype(np.float32) # оптимизирую типы данных для экономии RAM и логарифмирую в одну шкалу для удобства модели

df_long['lag_1'] = df_long.groupby('Page')['Visits'].shift(1)   # Вчера
df_long['lag_7'] = df_long.groupby('Page')['Visits'].shift(7)   # Неделю назад
df_long['lag_14'] = df_long.groupby('Page')['Visits'].shift(14) # Две недели назад
df_long['lag_21'] = df_long.groupby('Page')['Visits'].shift(21) # Три недели назад

df_long['day_of_week'] = df_long['Date'].dt.dayofweek
df_long['day_of_month'] = df_long['Date'].dt.day

df_long['rolling_mean_7'] = df_long.groupby('Page')['Visits'].transform(lambda x: x.rolling(window=7).mean())
df_long['rolling_std_7'] = df_long.groupby('Page')['Visits'].transform(lambda x: x.rolling(window=7).std())
df_long['rolling_max_7'] = df_long.groupby('Page')['Visits'].transform(lambda x: x.rolling(window=7).max())

df_long['diff_1_2'] = df_long['lag_1'] - df_long.groupby('Page')['Visits'].shift(2)
df_long['diff_1_7'] = df_long['lag_1'] - df_long['lag_7']
df_long = df_long.sort_values(by=['Page', 'Date'])



df_model = df_long.dropna().copy()
df_model['is_weekend'] = df_model['day_of_week'].isin([5, 6]).astype(np.int8)


def extract_features(df):
    # Разделяем строку с конца
    parts = df['Page'].str.split('_')
    df['agent'] = parts.str[-1].astype('category')      # spider или all-agents
    df['access'] = parts.str[-2].astype('category')     # desktop, mobile-web, all-access
    # Язык обычно идет перед .wikipedia.org
    df['language'] = parts.str[-3].str.split('.').str[0].astype('category')
    return df

df_model = extract_features(df_model)

split_date = '2019-11-30'
features = ['lag_1', 'lag_7', 'lag_14', 'lag_21', 
            'is_weekend', 'day_of_week', 'day_of_month', 
            'agent', 'access', 'language',
            'rolling_mean_7', 'rolling_std_7', 'diff_1_7',
            'page_median', 'page_std',
            'diff_1_2', 'rolling_max_7', 'is_high_traffic']
target = 'Visits'

train = df_model[df_model['Date'] <= split_date]
test = df_model[df_model['Date'] > split_date]

page_stats = train.groupby('Page')['Visits'].agg(['median', 'std']).reset_index()
page_stats.columns = ['Page', 'page_median', 'page_std']

train = train.merge(page_stats, on='Page', how='left')
test = test.merge(page_stats, on='Page', how='left')

q75 = train['page_median'].quantile(0.75)
train['is_high_traffic'] = (train['page_median'] > q75).astype(np.int8)
test['is_high_traffic'] = (test['page_median'] > q75).astype(np.int8)

train['page_std'] = train['page_std'].fillna(0)
test['page_std'] = test['page_std'].fillna(0)

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]


N_ESTIMATORS = 150
LRN = 0.04

import lightgbm as lgb

model = lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators=N_ESTIMATORS,
    learning_rate=LRN,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)


evals_result = {}


model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='mae',
    eval_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100),
        lgb.record_evaluation(evals_result)
    ]
)
joblib.dump(model, f'./media_ai/lgbm_traffic_model_{N_ESTIMATORS}_{LRN}.pkl')

preds_log = model.predict(X_test)
preds = np.expm1(preds_log)       # Прогноз в реальных числах
y_test_real = np.expm1(y_test)  

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

print(f"Ошибка SMAPE: {smape(y_test_real, preds):.2f}%")

mae = mean_absolute_error(y_test_real, preds)
print(f"Средняя абсолютная ошибка (MAE): {mae:.2f} просмотров")

lgb.plot_importance(model, importance_type='gain', figsize=(10, 6), title='Важность признаков (Feature Importance)')
plt.tight_layout()
plt.savefig(f"./media_ai/model_{N_ESTIMATORS}_{LRN}.png")

plt.figure(figsize=(12, 6))
plt.plot(np.expm1(y_test.values[:100]), label='Реальные (Actual)', alpha=0.7, linewidth=2)
plt.plot(preds[:100], label='Прогноз (Predicted)', alpha=0.7, linestyle='--')
plt.legend()
plt.title(f"Сравнение реальных данных и прогноза (MAE: {mae:.2f})")
plt.ylabel("Количество просмотров")
plt.grid(True, alpha=0.3)
plt.savefig(f"./final_media/real-vs-pred-{N_ESTIMATORS}-{LRN}.png")





plt.figure(figsize=(10, 6))

# Извлекаем данные (в LightGBM по умолчанию метрика l1 для MAE)
epochs = len(evals_result['train']['l1'])
x_axis = range(0, epochs)

plt.plot(x_axis, evals_result['train']['l1'], label='Обучение (Train)')
plt.plot(x_axis, evals_result['valid']['l1'], label='Валидация (Validation)')

plt.title('Динамика обучения (MAE Loss)')
plt.xlabel('Количество деревьев (Итерации)')
plt.ylabel('Ошибка MAE (в логарифмах)')
plt.legend()
plt.grid(True, alpha=0.3)

# Отмечаем точку остановки
best_iter = model.best_iteration_
plt.axvline(x=best_iter, color='r', linestyle='--', label='Лучшая итерация')

plt.savefig(f"./final_media/learning-curve-{N_ESTIMATORS}-{LRN}.png")


# График распределения ошибок
residuals = y_test_real - preds
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=100, kde=True)
plt.yscale('log') # Логарифмическая шкала для частоты
plt.title('Распределение остатков (Log Scale)')
plt.xlabel('Ошибка (Реальность - Прогноз)')
plt.ylabel('Частота (Log)')
plt.xlim(-500, 500) # Сужаем диапазон, чтобы рассмотреть центр
plt.grid(True, alpha=0.2)
plt.savefig(f"./final_media/residuals-hist-improved-{N_ESTIMATORS}-{LRN}.png")



# e_p = df_long['Page'].unique()[0]
# s_d = df_long[df_long['Page'] == e_p].tail(100) # за последние 100 дней



# plt.figure(figsize=(15, 6))
# plt.plot(s_d['Date'], s_d['Visits'], label='Текущий трафик', linewidth=2)
# plt.plot(s_d['Date'], s_d['lag_7'], label='Лаг 7 дней (прошлая неделя)', linestyle='--', alpha=0.7)
# plt.title(f'Как прошлая неделя предсказывает текущую\n(Страница: {e_p})')
# plt.legend()
# plt.savefig("./media/lag_7.png")

# corr_matrix = df_long[['Visits', 'lag_1', 'lag_7', 'lag_14', 'lag_21', 'lag_28', 'lag_30']].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0)
# plt.title('Сила связи между текущим трафиком и прошлым (Лагами)')
# plt.savefig("./media/corr_visits_periodic.png")


# plt.figure(figsize=(10, 6))
# sns.boxplot(x='day_of_week', y='Visits', data=df_long)
# plt.title('Распределение трафика по дням недели')
# plt.xlabel('День недели (0=Пн, 6=Вс)')
# plt.ylabel('Количество визитов')
# plt.ylim(0, df_long['Visits'].quantile(0.95))
# plt.grid(axis='y', alpha=0.3)
# plt.savefig("./media/review_days_activity_week.png")

# plt.figure(figsize=(10, 6))
# monthly_avg = df_long.groupby('day_of_month')['Visits'].mean()
# monthly_avg.plot(marker='o', color='tab:blue', linewidth=2)
# plt.title('Средний трафик по дням месяца')
# plt.xlabel('День месяца')
# plt.ylabel('Среднее кол-во визитов')
# plt.grid(True, alpha=0.3)
# plt.xticks(range(1, 32))
# plt.ylim(0, monthly_avg.max() * 1.1) 
# plt.savefig("./media/review_days_activity_month.png")
