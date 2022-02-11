import copy
import os
import pickle
import random

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import warnings

from CALCULATE_MODULE.GNN.my_dataset import get_dataset, _HOMO
from CALCULATE_MODULE.visulization.figure_helper import plot_fiting_points

warnings.filterwarnings('ignore')  # filter warnings
from pylab import mpl  # 正常显示中文

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示符号
from matplotlib import rcParams

rcParams['axes.unicode_minus'] = False
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor


SEED = 1423


def setup_seed(seed):
    global SEED
    SEED = seed
    if seed == None:
        return
    np.random.seed(SEED)
    random.seed(SEED)


def formatData(dataset_for_train_val, dataset_for_test, scale_x=None, scale_y=None):
    datasets = {
        "train": dataset_for_train_val,
        "test": dataset_for_test
    }
    X = {"train": list(), "test":list()}
    Y = copy.deepcopy(X)
    for type in ["train", "test"]:
        for graph,label in datasets[type]:
            features = list()  # [晶粒数量，晶粒各属性平均值，晶粒各属性最大值，晶粒各属性最小值，晶界数量，晶界长度平均值，晶界长度最大值，晶界长度最小值]
            # 晶粒数量
            # features.append(graph.number_of_nodes())
            #  晶粒属性（平均，最大，最小）
            all_nodes_features = graph.ndata['h']
            features += all_nodes_features.mean(0).numpy().tolist()
            features += all_nodes_features.max(0).values.numpy().tolist()
            features += all_nodes_features.min(0).values.numpy().tolist()

            # 晶界数量
            # features.append(graph.number_of_edges()/2)  #由于构建neo4j图的时候构建了双向连接
            # # 晶界长度(平均，最大，最小)
            all_edges_bl = graph.edata['bl']
            features.append(all_edges_bl.mean(0).numpy().tolist())
            features.append(all_edges_bl.max(0).values.numpy().tolist())
            features.append(all_edges_bl.min(0).values.numpy().tolist())
            # # 晶界的取向差特征
            all_nodes_mis = graph.edata['dis']
            features.append(all_nodes_mis.mean(0).numpy().tolist())
            features.append(all_nodes_mis.max(0).values.numpy().tolist())
            features.append(all_nodes_mis.min(0).values.numpy().tolist())

            X[type].append(features)
            Y[type].append(label.numpy())

    # 归一化
    # 注意：测试集的归一化和逆转应该使用训练集的均值和方差！！！！
    if scale_x:
        scale_x.fit(np.array(X["train"]))
        X["train"] = scale_x.transform(np.array( X["train"] ))
        X["test"]  = scale_x.transform(np.array( X["test"] ))
    if scale_y:
        scale_y.fit(np.array(Y["train"]))
        Y["train"] = scale_y.transform(np.array( Y["train"] ))
        Y["test"]  = scale_y.transform(np.array( Y["test"] ))

    return X['train'], X['test'], Y['train'], Y['test']


def train(dataset_for_train_val, dataset_for_test, scale_x=None, scale_y=None, k_fold=10, label_options=None):
    if k_fold == 1 :
        k_fold = len(dataset_for_train_val)
    # build models
    models = {
        "LR": LinearRegression(),
        "Ridge": Ridge(alpha=10, random_state=SEED),
        "SVR": MultiOutputRegressor(SVR(C=30, degree=1, gamma="auto", kernel="rbf")),
        "KNN": KNeighborsRegressor(leaf_size=10, n_neighbors=5, p=1, weights='distance'),
        "ET": MultiOutputRegressor(ExtraTreesRegressor(n_estimators=4, random_state=SEED)),
        "RF": MultiOutputRegressor(RandomForestRegressor(n_estimators=15, random_state=SEED)),
        "GBDT": MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
        "XGB": MultiOutputRegressor(XGBRegressor(random_state=SEED)),
        "MLP": MLPRegressor(hidden_layer_sizes=(200, 500, 200), max_iter=500, random_state=SEED)
    }
    # load data
    x_train, x_test, y_train, y_test = formatData(dataset_for_train_val, dataset_for_test, scale_x, scale_y)

    for model_name, model in models.items():
        ret = cross_validate(model, x_train, y_train, cv=k_fold, return_estimator=True,
                             scoring=['explained_variance', 'neg_mean_squared_error', 'neg_mean_absolute_error',
                                      'neg_root_mean_squared_error', 'r2'])
        val_mse = ret['test_neg_mean_squared_error'].mean()
        val_mae = ret['test_neg_mean_absolute_error'].mean()
        val_ev_score = ret['test_explained_variance'].mean()
        val_r2 = ret['test_r2'].mean()

        test_pred = np.array(([model.predict(x_test) for model in ret['estimator']])).mean(axis=0)
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_ev_score = explained_variance_score(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        print('{}:     \t mse     \t mae     \t ev      \t r2    \n'
              # '验证集： \t {:.3f}  \t {:.3f}  \t {:.3f}  \t {:.3f} \n'
              '测试集： \t {:.3f}  \t {:.3f}  \t {:.3f}  \t {:.3f}'.
              format(model_name,
                     # val_mse, val_mae, val_ev_score, val_r2,
                     test_mse, test_mae, test_ev_score, test_r2))

        test_pred_ = scale_y.inverse_transform(test_pred) if scale_y else test_pred
        y_test_    = scale_y.inverse_transform(y_test)    if scale_y else y_test
        print("预测值：", *("%10.2f" % o.item() for o in test_pred_.flatten()))
        print("真实值：", *("%10.2f" % l.item() for l in y_test_.flatten()), end="\n\n")

        figure = ""
        for op in label_options:
            figure += op[0]

        plot_fiting_points(np.ravel(test_pred), np.ravel(y_test),
                           title="R2 = %.3f" % test_r2,
                           savefile="{}/figures/{}({}).png".format(os.path.dirname(__file__), model_name, figure))
        # print(y_test )
        # print(np.mean(test_preds,0).tolist())


def run_task(graph_database = "DB3*3", label_options=["ys"]):

    scale_x = StandardScaler()
    scale_y = StandardScaler()
    # scale_x = None
    # scale_y = None
    try:
        dataset_for_train_val, dataset_for_test \
            = get_dataset(graph_type=_HOMO, graph_database=graph_database, test_size=1, target="block",
                          label_options=label_options, directions="90", force_reload=False ,
                           scaler_x=None, scaler_y=None, random_state=SEED, verbose=True)    # 注意这里面不需要特征标准化，而应该在当前程序中对新特征进行标准化
    except Exception:
        dataset_for_train_val, dataset_for_test \
            = get_dataset(graph_type=_HOMO, graph_database=graph_database, test_size=1, target="block",
                          label_options=label_options, directions="90", force_reload=True,
                          scaler_x=None, scaler_y=None, random_state=SEED, verbose=True)

    train(dataset_for_train_val, dataset_for_test, scale_x, scale_y, k_fold=10, label_options=label_options)

# 设置随机种子
setup_seed(1423)

if __name__ == '__main__':
    for labels in (["ys"], ["uts"], ["el" ], ["ys", "uts", "el" ]):
        print("-------------------------{}-------------------------".format(",".join(labels)))
        run_task(label_options = labels)
    # print(sklearn.metrics.SCORERS.keys())
