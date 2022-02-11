# -*- coding: utf-8 -*-
import numpy as np
from numpy import min as min,max as max
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from utils.file_util import create_file_dir_not_exist

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# matplotlib画图中中文显示会有问题，需要这两行设置默认字体

# 设置大小
SIZE = 16



def plot_fiting_points(preds, labels, label=None, title=None, savefile=None,
                                      min_axi=None, max_axi=None, x_label="label", y_label="prediction"):
    if isinstance(preds, Tensor): preds = preds.numpy()
    if isinstance(labels, Tensor): labels = labels.numpy()
    # 找出最小点 和 最大点
    if min_axi:
        plt.xlim(xmin=min_axi)
        plt.ylim(ymin=min_axi)
    if max_axi:
        plt.xlim(ymax=max_axi)
        plt.ylim(ymax=max_axi)

    # 标题
    if title: plt.title(title, fontsize=SIZE+4)

    # 坐标标签
    plt.xlabel(x_label, fontsize=SIZE)
    plt.ylabel(y_label, fontsize=SIZE)

    # 刻度
    plt.xticks(None, fontsize=SIZE)
    plt.yticks(None, fontsize=SIZE)

    # 散点图
    plt.scatter(labels, preds, s=80, c="#066592", alpha=0.4, label=label)
    if label: plt.legend(fontsize=SIZE) # 加图例

    # 画中间线
    white = (max([max(preds), max(labels)]) - min([min(preds), min(labels)])) / 3
    line_data = [min([min(preds), min(labels)]) - white,
                 max([max(preds), max(labels)]) + white]
    plt.plot(line_data,line_data, linewidth='1', color='#333333')

    # 保存图片
    if savefile:
        create_file_dir_not_exist(savefile)
        plt.savefig(savefile, dpi=300)

    plt.show()


if __name__ == '__main__':
    plot_fiting_points_for_less_label([23.46,23.20,23.38,23.60,38.33,38.13,38.34,38.52,37.22,38.53,38.81,38.55,21.68,21.86,21.95,21.89,22.00,22.34,22.39,21.84],
 [23.50,23.50,23.50,23.50,31.30,31.30,31.30,31.30,38.70,38.70,38.70,38.70,22.00,22.00,22.00,22.00,21.90,21.90,21.90,21.90],title="dsf 第三方", savefile=" das.png")