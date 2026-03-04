from datetime import datetime
import os
import kagglehub


dataset_dir = kagglehub.dataset_download("sandeshbhat/wikipedia-web-traffic-201819")

PATH = os.path.join(dataset_dir, "Wiki_Page_views.csv")
FEATURES = [
    'lag_1', 'lag_2', 'lag_7', 'lag_14', 'lag_21', 
    'is_weekend', 'day_of_week', 'day_of_month', 
    'rolling_mean_7', 'rolling_std_7', 'rolling_max_7',
    'z_7', 'slope_7',
    'diff_1_7', 'diff_1_2',
    'page_median', 'page_std',
    'global_mean',
    'agent', 'access', 'language',
]
CAT_FEATURES = ['agent', 'access', 'language']

SPLIT_DATE = '2019-11-30'
TARGET = 'Visits'

N_ESTIMATORS = 900
LRN = 0.09
DATE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
