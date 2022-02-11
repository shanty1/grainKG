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
from matplotlib import rcParams, pyplot as plt

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
        # "LR": LinearRegression(),
        "Ridge": Ridge(alpha=10, random_state=SEED),
        "SVR": MultiOutputRegressor(SVR(C=30, degree=1, gamma="auto", kernel="rbf")),
        "KNN": KNeighborsRegressor(leaf_size=10, n_neighbors=5, p=1, weights='distance'),
        # "ET": MultiOutputRegressor(ExtraTreesRegressor(n_estimators=4, random_state=SEED)),
        "RF": MultiOutputRegressor(RandomForestRegressor(n_estimators=15, random_state=SEED)),
        # "GBDT": MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
        # "XGB": MultiOutputRegressor(XGBRegressor(random_state=SEED)),
        "MLP": MLPRegressor(hidden_layer_sizes=(200, 500, 200), max_iter=500, random_state=SEED)
    }
    # load data
    x_train, x_test, y_train, y_test = formatData(dataset_for_train_val, dataset_for_test, scale_x, scale_y)

    colos=["#999999","#FF9900","#99CC66","#666699","#66CCCC","#FF6666"]
    for i, (model_name, model) in enumerate(models.items()):
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

        # 每个模型画一张图
        # plot_fiting_points(np.ravel(test_pred), np.ravel(y_test),
        #                    title="R2 = %.3f" % test_r2,
        #                    savefile="{}/figures/{}({}).png".format(os.path.dirname(__file__), model_name, figure))
        # 所有模型画在一张图里
        y_test_ = np.ravel(y_test_).tolist()
        test_pred_ = np.ravel(test_pred_).tolist()
        data_num = len(np.ravel(test_pred))
        plt.scatter(y_test_, test_pred_,label=model_name,
                    c=[colos[i]]*data_num, alpha=0.85,s=[50]*data_num)

    # 绘制gnn的散点
    if label_options == ["ys"]: #0.932
        xs = [124.00, 87.20, 63.10, 186.90, 128.90] # 预测值
        ys = [121.32, 63.43, 66.84, 183.29, 128.75] # 真实值
    elif label_options == ["uts"]: #0.947
        xs = [286.60, 262.10, 279.80, 341.20, 316.10]
        ys = [286.37, 268.14, 278.11, 328.38, 313.80]
    elif label_options == ["el"]: #0.929
        xs = [23.50, 35.30, 38.70, 22.00, 21.90]
        ys = [23.54, 31.90, 36.09, 21.83, 22.12]
    else:
        xs = [124.00, 286.60,  23.50,  87.20, 262.10,  31.30,  63.10, 279.80,  38.70, 186.90, 341.20,  22.00, 128.90, 316.10,  21.90]
        ys = [123.91, 286.19,  23.46,  75.90, 268.72,  34.61,  74.12, 271.87,  35.30, 178.70, 335.37,  22.67, 129.46, 315.28,  22.10]

    plt.scatter(xs, ys, label="HGGAT",
                c=[colos[i+1]] * data_num, alpha=0.85, s=[50] * data_num)
    # 绘制拟合线
    min = np.min(y_test_ + test_pred_ + xs)
    max = np.max(y_test_ + test_pred_ + xs)
    plt.plot([min, max], [min, max], color="grey", linewidth=1.5, linestyle='--')

    plt.tick_params(labelsize=15)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.5)

    plt.savefig("{}/figures/{}.png".format(os.path.dirname(__file__), figure), dpi=300, transparent=True)
    plt.show()
        # print(y_test )
        # print(np.mean(test_preds,0).tolist())


def run_task(graph_database = "DB1*1", label_options=["ys"]):

    scale_x = StandardScaler()
    scale_y = StandardScaler()
    # scale_x = None
    # scale_y = None
    try:
        dataset_for_train_val, dataset_for_test \
            = get_dataset(graph_type=_HOMO, graph_database=graph_database, test_size=0.3, target="block",
                          label_options=label_options, directions="90", force_reload=False ,
                           scaler_x=None, scaler_y=None, random_state=SEED, verbose=True)    # 注意这里面不需要特征标准化，而应该在当前程序中对新特征进行标准化
    except Exception:
        dataset_for_train_val, dataset_for_test \
            = get_dataset(graph_type=_HOMO, graph_database=graph_database, test_size=0.3, target="block",
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
