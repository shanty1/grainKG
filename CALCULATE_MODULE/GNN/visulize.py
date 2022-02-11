import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from CALCULATE_MODULE.visulization.feat_vis import FtVisual

SEED=1

def show3Ddistribution(X, labels=None,figsiz=(8, 6), elev=48, azim=134, show_ticks=True, ticks=None,show_ticklabels=False):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if labels is not None and not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    fig = plt.figure(figsize=figsiz)  # 构建画布
    plt.clf()  # 清除画布图像
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)  # 构建3D画布
    plt.cla()  # 清除轴
    # 绘制数据点的label, 在降维后的3维数据的中心点展示label
    if labels is not None:
        for index, label in enumerate(set(labels)):
            # label == labels 找出labels中label的所有下标
            ax.text3D(X[label == labels, 0].mean(),  # 第一维坐标
                      X[label == labels, 1].mean(),  # 第二维坐标，偏移1.5，省得挡到数据
                      X[label == labels, 2].mean() + index * 0.5,  # 第三维坐标
                      label,  # 展示的标签名
                      horizontalalignment='center',
                      bbox=dict(alpha=0.8, edgecolor='w', facecolor='w'))

    # Reorder the labels to have colors matching the cluster results
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=30, marker="o",
               alpha=0.8, edgecolor='k', cmap="jet")
    ax.set_title("节点特征可视化")
    plt.legend(labels=['not at all', 'a small doses', 'a large doses'])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # 现实坐标轴刻度和刻度线
    if not show_ticks:
        # 设置刻度值，递增数组，绘图点会根据刻度值自动调整坐标位置
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    elif ticks is not None:
        ax.set_xticks(ticks[0])
        ax.set_yticks(ticks[1])
        ax.set_zticks(ticks[2])
    # 显示坐标轴标签
    if not show_ticklabels:
        # 设置标签替代刻度值，数组长度必须与刻度数组一致
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
    plt.show()
    plt.close()
    print("draw done")

def pca(x, n_components=3, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=SEED):
    data =  PCA(n_components=n_components,copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state).fit_transform(x)
    return (data[:,i] for i in range(n_components))

def tsne(x, n_components=3, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-7,
             metric="euclidean", init="random", verbose=0, random_state=SEED, method='barnes_hut', angle=0.5, n_jobs=None):
    data = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress, min_grad_norm=min_grad_norm,metric=metric, init=init, verbose=verbose, random_state=random_state, method=method, angle=angle, n_jobs=n_jobs).fit_transform(x)
    return (data[:,i] for i in range(n_components))

def draw2DAll():
    df = pd.read_csv('result_save/graph_result1.csv')
    feature = df.drop(columns=['label', 'belong']).values
    label = df.loc[:, 'label'].values
    x,y,z = pca(feature,n_components=3)

def draw3D():
    df = pd.read_csv('result_save/graph_result1.csv')
    df.sort_values(axis=0, by='label', inplace=True)
    subgraphs = list()
    for id in df['belong'].values:
        if id not in subgraphs:  subgraphs.append(id)

    figure = plt.figure(figsize=(20, 30))  # create a figure
    row_num = math.ceil((len(subgraphs) + 1) / 3)
    figure.suptitle("TSNE Feature Distributions")

    for i, subgraph_name in enumerate(subgraphs):
        subgraph = df.loc[df['belong'] == subgraph_name]
        feature = subgraph.drop(columns=['label', 'belong']).values
        label = subgraph.loc[:, 'label'].values
        x, y, z = pca(feature, n_components=3)
        ax = figure.add_subplot(row_num, 3, i + 1, projection='3d')
        # ax.set_ylim(-2.5, 5)
        # ax.set_zlim([-2.5, 10])
        # ax.set_xlim([-5, 5])
        ax.set_title(label[0])
        ax.scatter(x, y, z)

    ax = figure.add_subplot(row_num, 3, len(subgraphs) + 1, projection='3d')
    ax.set_title("All Feature")
    labels = set(df.label.values)
    for i,label_name in enumerate(labels):
        sublabelgraph = df.loc[df['label'] == label_name]
        feature = sublabelgraph.drop(columns=['label', 'belong']).values
        x, y, z = pca(feature, n_components=3)
        ax.scatter(x, y, z, label=label_name)
    ax.legend(loc="upper right",  bbox_to_anchor=(0, 1),ncol=1)
    # plt.legend(loc="best")
    plt.show()


def AXI3D_label():
    def show(df):
        ftV = FtVisual(elev=200, azim=200, show_ticklabels=True,
                       figsiz=(12, 9),
                       ticks=[[-10, -5, 0, 5, 10],
                              [-5, 0, 5, 10, 15, 20, 25],
                              [-4, -2, 0, 2, 4, 6, 8]]
                       )
        features = df.drop(columns=['label', 'belong']).values
        labels = df.loc[:, 'label'].values
        X = ftV.fit_pca(features)
        ftV.show3Ddistribution(X, labels)

    df = pd.read_csv('result_save/graph_result1.csv')
    show(df)
    for l in set(df.label.values):
        show(df.loc[df['label'] == l])

if __name__ == '__main__':
    # AXI3D_label()
    draw3D()