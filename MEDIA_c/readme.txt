добавил новые признаки z_7, global_mean, slope_7, lag_2
переписал diff_1_2
переписал метод семплирования данных уже не по индексам строк, а по самим объектам, чтоб разрывов не было
метод для создания столбца slope_7 был написан на C.

[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.140341 seconds. 
You can set `force_row_wise=true` to remove the overhead. 
And if memory is not enough, you can set `force_col_wise=true`. 
[LightGBM] [Info] Total Bins 3882 
[LightGBM] [Info] Number of data points in the train set: 13560000, number of used features: 21 
[LightGBM] [Info] Start training from score 4.704485 
Training until validation scores don't improve for 50 rounds 
Did not meet early stopping. 
Best iteration is: 
[900]   train's rmse: 0.0320634   train's l2: 0.00102806    valid's rmse: 0.0371877  valid's l2: 0.00138293 





Ошибка SMAPE: 12.03640% 
Средняя абсолютная ошибка (MAE): 281.61893 
Средняя квадратичная ошибка (MSE): 794417439.55323 
Корневая средняя квадртичная ошибка (RMSE): 28185.41182 
Коэффициент Детерминации (R^2): 92.38910% 





Min: -4967623.567167245 
Max: 11478555.311321372 
Std: 28185.42486900197 
95 percentile: 30.194002314530266 
99 percentile: 226.74486238539663 
[-224.7159182 226.74486239]
