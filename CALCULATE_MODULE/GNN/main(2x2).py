import _thread
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

from CALCULATE_MODULE.GNN.model import HGAT_DNN
from CALCULATE_MODULE.GNN.my_dataset import get_dataset, _HETERO
from CALCULATE_MODULE.GNN.trainer import Trainer

SEED = 1423

def setup_seed(seed):
    global SEED
    SEED = seed
    if seed == None:
        return
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

def build_trainer(name, update_attr_nodes, readout, residual, label_options, scale_x = None, scale_y=None,
                  graph_database="DB2*2", test_size=1, target="block"):

    # 数据集
    try:
        dataset_for_train_val, dataset_for_test \
            = get_dataset(graph_type=_HETERO, graph_database=graph_database, test_size=test_size, target=target,
                          label_options=label_options, directions="90", force_reload=False,
                          self_loop=True, scaler_x=scale_x, scaler_y=scale_y,
                          update_attr_nodes=update_attr_nodes, random_state=SEED, verbose=True, )
    except Exception:
        dataset_for_train_val, dataset_for_test \
            = get_dataset(graph_type=_HETERO, graph_database=graph_database, test_size=test_size, target=target,
                          label_options=label_options, directions="90", force_reload=True,
                          self_loop=True, scaler_x=scale_x, scaler_y=scale_y,
                          update_attr_nodes=update_attr_nodes, random_state=SEED, verbose=True, )

    out_dim = len(dataset_for_train_val.labels[0])
    model = HGAT_DNN(hidden_dim=80, out_dim=out_dim, num_heads=1, num_cov_layer=3, num_nn_layer=3,
                     activation=F.relu, residual=residual, readout=readout,
                     feat_drop=0., attn_drop=0., nn_drop=0., negative_slope=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.65)

    trainer = Trainer(model_name = name,
                       model = model,
                       trainset = dataset_for_train_val,
                       test_set = dataset_for_test,
                       optimizer = optimizer,
                       scheduler = scheduler,
                       random_state=SEED)
    return trainer


"""
    【作者提示】：
        1. 对比训练请固定随机种子$SEED$，并不要更换。随机种子将影响数据集的划分、模型参数训练。
        2. 由于数据量小，使用留一法训练能达到最大的拟合性能。
"""

""" 预测样品块的子图 """
def main1():
    # 设置随机数种子
    setup_seed(1423)

    trainers = [
                     # name          update_attr_nodes   readout     residual                                             # seed =  1423
        # build_trainer("trainer3g-1(norm)(2X2)",    False,              False,      False, ["ys","uts","el"], None,StandardScaler()),  #
        build_trainer("trainer3g-2(norm)(2X2)",    False,              False,      True,  ["ys","uts","el"], None,StandardScaler() ), #
        build_trainer("trainer3g-3(norm)(2X2)",    True,               False,      True,  ["ys","uts","el"], None,StandardScaler() ), #
        build_trainer("trainer8g-4(norm)(2X2)",    True,               True,       True,  ["ys","uts","el"], None,StandardScaler() ), #
        build_trainer("trainer8g-5(norm)(2X2)",    False,              True,       True,  ["ys","uts","el"], None,StandardScaler() ), #
        # build_trainer("trainer3g-6(norm)(2X2)",    False,              True,       False, ["ys","uts","el"], None,StandardScaler()),  #
        # #
        # build_trainer("trainer3g.y-1(norm)(2X2)", False, False, False, "ys", None,StandardScaler()),                                  #
        build_trainer("trainer3g.y-2(norm)(2X2)", False, False, True,  "ys", None,StandardScaler()),                                  #
        build_trainer("trainer3g.y-3(norm)(2X2)", True,  False, True,  "ys", None,StandardScaler()),                                  #
        build_trainer("trainer8g.y-4(norm)(2X2)", True,  True,  True,  "ys", None,StandardScaler()),                                  #
        build_trainer("trainer8g.y-5(norm)(2X2)", False, True,  True,  "ys", None,StandardScaler()),                                  #
        # #
        # build_trainer("trainer3g.u-1(norm)(2X2)", False, False, False, "uts", None,StandardScaler()),                                  #
        build_trainer("trainer3g.u-2(norm)(2X2)", False, False, True,  "uts", None,StandardScaler()),                                  #
        build_trainer("trainer3g.u-3(norm)(2X2)", True,  False, True,  "uts", None,StandardScaler()),                                  #
        build_trainer("trainer8g.u-4(norm)(2X2)", True,  True,  True,  "uts", None,StandardScaler()),                                  #
        build_trainer("trainer8g.u-5(norm)(2X2)", False, True,  True,  "uts", None,StandardScaler()),                                  #
        #
        # build_trainer("trainer3g.e-1(norm)(2X2)", False, False, False, "el", None,StandardScaler()),                                   #
        build_trainer("trainer3g.e-2(norm)(2X2)", False, False, True,  "el", None,StandardScaler()),                                   #
        build_trainer("trainer3g.e-3(norm)(2X2)", True,  False, True,  "el", None,StandardScaler()),                                   #
        build_trainer("trainer8g.e-4(norm)(2X2)", True,  True,  True,  "el", None,StandardScaler()),                                   #
        build_trainer("trainer8g.e-5(norm)(2X2)", False, True,  True,  "el", None,StandardScaler()),                                   #
    ]

    for trainer in trainers:                 # epoch  k    batch
        # _thread.start_new_thread(trainer.train, (300, 10, 10))
        # trainer.train (200, 10)
        pass

    for trainer in trainers:
        trainer.test()

if __name__ == '__main__':
    main1()