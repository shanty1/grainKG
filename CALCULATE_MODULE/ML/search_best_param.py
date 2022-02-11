import random

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import warnings

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from CALCULATE_MODULE.GNN.my_dataset import get_dataset, _HOMO
from CALCULATE_MODULE.ML.main import formatData

warnings.filterwarnings('ignore') # filter warnings
from pylab import mpl # 正常显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示符号
from matplotlib import rcParams
rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


SEED = None

def setup_seed(seed):
    global SEED
    SEED = seed
    if seed == None:
        return
    np.random.seed(SEED)
    random.seed(SEED)


def train(dataset_for_train_val, dataset_for_test, scale_x=None, scale_y=None, k_fold=10):
    scoring = ['explained_variance', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    # build models

    gridcv_list = {
        'lr':    LinearRegression(),

        'Ridge': GridSearchCV(estimator=Ridge(),
                            cv=k_fold, n_jobs=-1, scoring=scoring, refit='neg_mean_absolute_error',
                            param_grid={
                                "alpha": np.logspace(-3, 2, 10)
        }),

        'knn':GridSearchCV(estimator=KNeighborsRegressor(),
                           cv=k_fold, n_jobs=-1, scoring=scoring, refit='neg_mean_absolute_error',
                           param_grid={
                               'n_neighbors': [nb for nb in range(5, 10)],
                               'p': [p for p in range(1, 3)],
                               'weights': ['uniform', 'distance'],
                               'leaf_size': [s for s in range(10, 15)],
        }),

        'svr': GridSearchCV(estimator=(SVR()),
                            cv=k_fold, n_jobs=-1, scoring=scoring, refit='neg_mean_absolute_error',
                            param_grid={
                                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                'degree': [i for i in range(1, 10)],
                                'gamma': ['scale', 'auto'],
                                'C': np.logspace(0, 3, 3),
        }),

        'et': GridSearchCV(estimator=(ExtraTreesRegressor()),
                           cv=k_fold, n_jobs=-1, scoring=scoring, refit='neg_mean_absolute_error',
                           param_grid=[
                               {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                               {'n_estimators': [3, 10], 'max_features': [2, 3, 4], 'bootstrap': [False], },
        ]),

        'rf': GridSearchCV(estimator=(RandomForestRegressor()),
                           cv=k_fold, n_jobs=-1, scoring=scoring, refit='neg_mean_absolute_error',
                           param_grid=[
                                {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                                {'n_estimators': [3, 10],     'max_features': [2, 3, 4],    'bootstrap': [False], },
        ]),

        'gbdt': GridSearchCV(estimator=(GradientBoostingRegressor()),
                            cv=k_fold, n_jobs=-1, scoring=scoring, refit='neg_mean_absolute_error',
                            param_grid={
                                'n_estimators': range(80, 200, 20),
                                'max_depth': range(2, 15, 2),
                                'learning_rate': [0.1, 0.01, 0.001],
                                'subsample': np.linspace(0.7, 0.9, 10),
        }),

        'xgboost': GridSearchCV(estimator=(XGBRegressor()),
                            cv=k_fold, n_jobs=-1, scoring=scoring, refit='neg_mean_absolute_error',
                            param_grid={
                                'n_estimators': range(10, 140, 20),
                                'max_depth': range(2, 15, 3),
                                'learning_rate':  [0.1, 0.05],
                                'subsample': np.linspace(0.7, 0.9, 2),
                                'colsample_bytree': np.linspace(0.5, 0.98, 3),
                                'min_child_weight': range(1, 9, 2)
        }),

        'mlp':GridSearchCV(estimator=MLPRegressor(),
                           cv=k_fold, n_jobs=-1, scoring=scoring, refit='neg_mean_absolute_error',
                            param_grid={
                                'hidden_layer_sizes':[(100, 250, 250, 100),
                                                      (100, 200, 200,200,100),
                                                      (100, 500, 500,200,100),
                                                      (100, 100, 100),
                                                      (100, 180, 180,180,180,50)]
        }),

    }

    # load data
    x_train, x_test, y_train, y_test = formatData(dataset_for_train_val, dataset_for_test, scale_x, scale_y)
    y_train, y_test = np.ravel(y_train), np.ravel(y_test)
    # train
    for model_name, gridcv in gridcv_list.items():
        print('\n', model_name)
        gridcv.fit(x_train, y_train)
        if isinstance(gridcv, GridSearchCV):
            print("best score:", gridcv.best_score_)
            print("best param:", gridcv.best_params_)

        test_pred = gridcv.predict(x_test)
        test_mse = mean_squared_error(y_test,test_pred)
        test_mae = mean_absolute_error(y_test,test_pred)
        test_ev_score = explained_variance_score(y_test,test_pred)
        test_r2 = r2_score(y_test,test_pred)
        print('{}:     \t mse     \t mae     \t ev      \t r2    \n'
              # '验证集： \t {:.3f}  \t {:.3f}  \t {:.3f}  \t {:.3f} \n'
              '测试集： \t {:.3f}  \t {:.3f}  \t {:.3f}  \t {:.3f} \n'.
              format(model_name,
                     # val_mse, val_mae, val_ev_score, val_r2,
                     test_mse, test_mae, test_ev_score, test_r2))

# 设置随机种子
setup_seed(1423)
if __name__ == '__main__':

        graph_database = "DB2*2"
        label_options = ["ys"]

        scale_x = StandardScaler()
        scale_y = StandardScaler()
        # scale_x = None
        # scale_y = None

        try:
            dataset_for_train_val, dataset_for_test \
                = get_dataset(graph_type=_HOMO, graph_database=graph_database, test_size=0.2, target="block",
                              label_options=label_options, directions="90", force_reload=False,
                              scaler_x=None, scaler_y=None, random_state=SEED,
                              verbose=True)  # 注意这里面不需要特征标准化，而应该在当前程序中对新特征进行标准化
        except Exception:
            dataset_for_train_val, dataset_for_test \
                = get_dataset(graph_type=_HOMO, graph_database=graph_database, test_size=0.2, target="block",
                              label_options=label_options, directions="90", force_reload=True,
                              scaler_x=None, scaler_y=None, random_state=SEED,
                              verbose=True)

        train(dataset_for_train_val, dataset_for_test, scale_x, scale_y, k_fold=10)
