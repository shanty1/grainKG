import copy

import torch
import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

from CALCULATE_MODULE.GNN import model, repository
from CALCULATE_MODULE.GNN import 结果还原1
from CALCULATE_MODULE.visulization.feat_vis import FtVisual
from sklearn.preprocessing import StandardScaler
import util
import dgl
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error,explained_variance_score,mean_absolute_error


ftV = FtVisual(elev=200, azim=200,show_ticklabels=True,
                figsiz=(12,9),
               # ticks=[[-15,-10,-5,0,5,10,15],
               #        [-10,-5,0,5,10,15,20],
               #        [-7.5,-5,-2.5,0,2.5,5,7.5,10]]
               )


def predict(labels = ["DB1*1", "Grain", "Sample5", "Block2", "Sub1"], return_feat=False):
    graph = repository.heterograph_feature(labels, update_attr_nodes=True)  #update_attr_nodes必须跟下面相同
    # 这里的train需要到main中去复制过来，保证参数一致   upd_node  readout  residual
    trainer = 结果还原1.build_trainer("trainer3g-5(norm)(1x1)", False, True, True, ["ys", "uts", "el"], None, StandardScaler(),
                  graph_database="DB1*1", test_size=0.3, target="block")  #

    out = trainer.predict(graph, return_feat=return_feat)
    return out

def vis3d(db="DB1*1", sample=1, block=1, n_components=3, combine=False):
    Xs = []
    for sub_id in range(1, eval(db.replace("DB",""))+1):
        labels = [db, "Grain", "Sample%d" % sample, "Block%d" % block, "Sub%d"%sub_id]
        ftV = FtVisual(elev=200, azim=200, show_ticklabels=True,
                       figsiz=(12, 9),
                       # ticks=[[-10, -5, 0, 5, 10],
                       #        [-5, 0, 5, 10, 15, 20, 25],
                       #        [-4, -2, 0, 2, 4, 6, 8]]
                       )
        Xs.append(predict(labels, True).squeeze(0))
    if combine:
        Xs = [torch.cat(Xs)]
    for x in Xs:
        x = ftV.fit_umap(x, n_components)
        if n_components == 2:
            plt.scatter(x[:, 0], x[:, 1], c=None, cmap='Spectral', s=5)
            plt.title("样品{}.{}".format(sample, block))
        else:
            ftV.show3Ddistribution(x,title="样品{}.{}".format(sample, block))
    plt.show()


if __name__ == '__main__':
    # vis3d(db="DB1*1", sample=1, block=1, n_components=3, combine=False)
    # vis3d(db="DB1*1", sample=2, block=1, n_components=3, combine=False)
    # vis3d(db="DB1*1", sample=3, block=1, n_components=3, combine=False)
    # vis3d(db="DB1*1", sample=4, block=1, n_components=3, combine=False)
    # vis3d(db="DB1*1", sample=5, block=1, n_components=3, combine=False)
    # vis3d(db="DB1*1", sample=6, block=1, n_components=3, combine=False)
    vis3d(db="DB1*1", sample=5, block=1, n_components=3, combine=True)
    vis3d(db="DB1*1", sample=5, block=2, n_components=3, combine=True)
    vis3d(db="DB1*1", sample=5, block=3, n_components=3, combine=True)

    vis3d(db="DB1*1", sample=6, block=1, n_components=3, combine=True)
    vis3d(db="DB1*1", sample=6, block=2, n_components=3, combine=True)
    vis3d(db="DB1*1", sample=6, block=3, n_components=3, combine=True)

    vis3d(db="DB1*1", sample=7, block=1, n_components=3, combine=True)
    vis3d(db="DB1*1", sample=7, block=2, n_components=3, combine=True)
    vis3d(db="DB1*1", sample=7, block=3, n_components=3, combine=True)
    # vis3d(db="DB1*1", sample=8, block=1, n_components=3, combine=False)
    #
    # digits = load_digits()
    #
    # mapper = umap.UMAP().fit(digits.data)
    # points(mapper, labels=digits.target)