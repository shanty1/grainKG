
""" 预测整个样品块 """
import os
import random
from pathlib import Path

import numpy as np
import torch
from dgl.nn.pytorch import GATConv
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
import torch.nn.functional as F

from CALCULATE_MODULE.GNN import config
from CALCULATE_MODULE.GNN.main import setup_seed, build_trainer
from CALCULATE_MODULE.GNN.model import HGAT_DNN, HomoGAT
from CALCULATE_MODULE.GNN.my_dataset import get_dataset, _HETERO, _HOMO
from CALCULATE_MODULE.GNN.repository import feature_props
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
                  graph_database="DB1*1", test_size=0.3, target="block"):

    # 数据集
    try:
        dataset_for_train_val, dataset_for_test \
            = get_dataset(graph_type=_HOMO, graph_database=graph_database, test_size=test_size, target=target,
                          label_options=label_options, directions="90", force_reload=False,
                          self_loop=True, scaler_x=scale_x, scaler_y=scale_y,
                          update_attr_nodes=update_attr_nodes, random_state=SEED, verbose=False, )
    except Exception:
        dataset_for_train_val, dataset_for_test \
            = get_dataset(graph_type=_HOMO, graph_database=graph_database, test_size=test_size, target=target,
                          label_options=label_options, directions="90", force_reload=True,
                          self_loop=True, scaler_x=scale_x, scaler_y=scale_y,
                          update_attr_nodes=update_attr_nodes, random_state=SEED, verbose=True, )

    out_dim = len(dataset_for_train_val.labels[0])

    model = HomoGAT(in_dim=len(feature_props), hidden_dim=20, out_dim=out_dim, heads=[1,1,1],
                    num_cov_layer=3, num_nn_layer=1, readout=readout,
                     activation=F.relu, residual=residual,
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
                       random_state=SEED,
                      save_path=config.pth_path + os.sep + Path(os.path.basename(__file__)).stem  # 重新训练时注释掉这条
                    )
    return trainer

if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(1423)

    trainers = [
                     # name          update_attr_nodes   readout     residual                                             # seed =  1423
        # build_trainer("homotrainer3g-1(norm)(1x1)",    False,  False, False, ["ys","uts","el"], None,StandardScaler()),  #
        # build_trainer("homotrainer3g-2(norm)(1x1)",    False,  False, True,  ["ys","uts","el"], None,StandardScaler()), #
        # build_trainer("homotrainer3g-3(norm)(1x1)",    True,   False, True,  ["ys","uts","el"], None,StandardScaler()), #
        # build_trainer("homotrainer3g-4(norm)(1x1)",    True,   True,  True,  ["ys","uts","el"], None,StandardScaler()), #
        # build_trainer("homotrainer3g-4(norm)(1x1)2",    True,   True,  True,  ["ys","uts","el"], None,StandardScaler()), #
        # build_trainer("homotrainer3g-4(norm)(1x1)3",    True,   True,  True,  ["ys","uts","el"], None,StandardScaler()), #
        # build_trainer("homotrainer3g-4(norm)(1x1)4",    True,   True,  True,  ["ys","uts","el"], None,StandardScaler()), #
        # build_trainer("homotrainer3g-4(norm)(1x1)5",    True,   True,  True,  ["ys","uts","el"], None,StandardScaler()), #
        # build_trainer("homotrainer3g-5(norm)(1x1)2",    False,  True,  True,  ["ys","uts","el"], None,StandardScaler()), #
        # build_trainer("homotrainer3g-6(norm)(1x1)",    False,  True,  False, ["ys","uts","el"], None,StandardScaler()),  #
        # #
        # build_trainer("homotrainer3g.y-1(norm)(1x1)", False, False, False, "ys", None,StandardScaler()), #
        # build_trainer("homotrainer3g.y-2(norm)(1x1)", False, False, True,  "ys", None,StandardScaler()), #
        # build_trainer("homotrainer3g.y-3(norm)(1x1)", True,  False, True,  "ys", None,StandardScaler()), #
        build_trainer("homotrainer3g.y-4(norm)(1x1)1", True,  True,  True,  "ys", None,StandardScaler()), #
        build_trainer("homotrainer3g.y-4(norm)(1x1)2", True,  True,  True,  "ys", None,StandardScaler()), #
        build_trainer("homotrainer3g.y-4(norm)(1x1)3", True,  True,  True,  "ys", None,StandardScaler()), #
        # build_trainer("homotrainer3g.y-5(norm)(1x1)", False, True,  True,  "ys", None,StandardScaler()), #
        # # #
        # build_trainer("homotrainer3g.u-1(norm)(1x1)", False, False, False, "uts", None,StandardScaler()), #
        # build_trainer("homotrainer3g.u-2(norm)(1x1)", False, False, True,  "uts", None,StandardScaler()), #
        # build_trainer("homotrainer3g.u-3(norm)(1x1)", True,  False, True,  "uts", None,StandardScaler()), #
        build_trainer("homotrainer3g.u-4(norm)(1x1)1", True,  True,  True,  "uts", None,StandardScaler()), #
        build_trainer("homotrainer3g.u-4(norm)(1x1)2", True,  True,  True,  "uts", None,StandardScaler()), #
        build_trainer("homotrainer3g.u-4(norm)(1x1)3", True,  True,  True,  "uts", None,StandardScaler()), #
        # build_trainer("homotrainer3g.u-5(norm)(1x1)", False, True,  True,  "uts", None,StandardScaler()), #
        # #
        # build_trainer("homotrainer3g.e-1(norm)(1x1)", False, False, False, "el", None,StandardScaler()),  #
        # build_trainer("homotrainer3g.e-2(norm)(1x1)", False, False, True,  "el", None,StandardScaler()),  #
        # build_trainer("homotrainer3g.e-3(norm)(1x1)", True,  False, True,  "el", None,StandardScaler()),  #
        # build_trainer("homotrainer3g.e-4(norm)(1x1)1", True,  True,  True,  "el", None,StandardScaler()),  #0.913
        build_trainer("homotrainer3g.e-4(norm)(1x1)2", True,  True,  True,  "el", None,StandardScaler()),  #
        build_trainer("homotrainer3g.e-4(norm)(1x1)3", True,  True,  True,  "el", None,StandardScaler()),  #
        build_trainer("homotrainer3g.e-4(norm)(1x1)4", True,  True,  True,  "el", None,StandardScaler()),  #
        # build_trainer("homotrainer3g.e-5(norm)(1x1)", False, True,  True,  "el", None,StandardScaler()),  #
    ]

    # for trainer in trainers:
    #     trainer.train(100, 10)
    for trainer in trainers:
        trainer.test()
