import numpy as np
import matplotlib.pyplot as plt
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 固定随机种子
SEED = 1423
np.random.seed(SEED)

class FtVisual():
    def __init__(self,figsiz=(8, 6), elev=48, azim=134, show_ticks=True, ticks=None,show_ticklabels=False):
        self.figsiz = figsiz
        self.show_ticklabels = show_ticklabels
        self.show_ticks = show_ticks
        self.ticks = ticks
        self.elev = elev
        self.azim = azim

    def show3Ddistribution(self, X, labels=None, title=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if labels is not None and not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        fig = plt.figure(figsize=self.figsiz) #构建画布
        plt.clf() # 清除画布图像
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=self.elev, azim=self.azim) # 构建3D画布
        plt.cla()  # 清除轴
        # 绘制数据点的label, 在降维后的3维数据的中心点展示label
        if labels is not None:
            for index,label in enumerate(set(labels)):
                # label == labels 找出labels中label的所有下标
                ax.text3D(X[label == labels, 0].mean(),  # 第一维坐标
                          X[label == labels, 1].mean(),  # 第二维坐标，偏移1.5，省得挡到数据
                          X[label == labels, 2].mean()+index*0.5,  # 第三维坐标
                          label,  # 展示的标签名
                          horizontalalignment='center',
                          bbox=dict(alpha=0.8, edgecolor='w', facecolor='w'))

        # Reorder the labels to have colors matching the cluster results
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=80, marker="o",
           alpha=0.1, edgecolor='k', cmap="jet")
        ax.set_title(title)
        plt.legend(labels=['not at all', 'a small doses', 'a large doses'])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # 现实坐标轴刻度和刻度线
        if not self.show_ticks:
            # 设置刻度值，递增数组，绘图点会根据刻度值自动调整坐标位置
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        elif self.ticks is not None:
            ax.set_xticks(self.ticks[0])
            ax.set_yticks(self.ticks[1])
            ax.set_zticks(self.ticks[2])
        # 显示坐标轴标签
        if not self.show_ticklabels:
            # 设置标签替代刻度值，数组长度必须与刻度数组一致
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
        plt.show()
        plt.close()
        print("draw done")

    def fit_tsne(self, x , n_components=3):
        return TSNE(n_components=n_components, random_state=SEED, n_iter=10000,learning_rate=100).fit_transform(x)

    def fit_pca(self,x, n_components=3):
        return  PCA(n_components=n_components, random_state=SEED).fit_transform(x)

    def fit_umap(self,x, n_components=3):
        return umap.UMAP(n_components=n_components, random_state=SEED).fit_transform(x)

if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    ftV = FtVisual(elev=48, azim=34)
    # x = ftV.fit_tsne(x)
    x = ftV.fit_pca(x)
    ftV.show3Ddistribution(x)
