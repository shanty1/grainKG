import random

import numpy as np
import py2neo
import torch
from torch import Tensor

''' # 文件明说 -- 记录数据的标签、训练&测试集的划分
数据说明：
  1.样品标签
    - Sample{i}：每个样品带有一个sample{编号}的标签，每一个sample{编号}对应同一属性值。（编号从1开始）
    - Block{j}：每个样品可能会被话分成n块，因此样品还有个No.{j},j=1,2,...,n标签
  2.图标签(下面三个标签确定一张子图)
    - Sample{i}：每个样品带有一个sample{编号}的标签，每一个sample{编号}对应同一属性值。（编号从1开始）
    - Block{j}：每个样品可能会被话分成n块，因此样品还有个No.{j},j=1,2,...,n标签
    - No.{j}：每个样品可能会被话分成n份，因此样品还有个No.{j},j=1,2,...,n标签
  3.节点标签类别
    - Grain：指晶粒节点
    - Size： 指大小节点
    - Ori：  指取向节点
'''

'''
    样品切块的数量（一个样品分为了几块）
    key:样品编号
    value：块数量
'''
blocks_num_dict = {"1": 1, "2": 1, "3": 1, "4": 5, "5": 5, "6": 6, "7": 5, "8": 5, }

# 样品对应性能数据
# 每个样品都有三个方向的测试性能，样品的力学形变方向见《力学性能数据》表
prop = {
    "ys": {  # 0°        45°       90°
        "1": {"0": 168.0, "45": 189.5, "90": 224.4},  # 1号 AZ31 挤压原始态
        "2": {"0": 66.7, "45": 87.0, "90": 198.2},  # 2号 AZ31 8.5%PRC + 退火
        "3": {"0": 142.1, "45": 146.9, "90": 154.6},  # 3号 AZ31 8.5%PRC + 13.5%PRT + 退火
        "4": {"0": 105.1, "45": 89.5, "90": 124.0},  # 4号 Mg-2Zn
        "5": {"0": 108.2, "45": 86.3, "90": 87.2},  # 5号 Mg-2Zn-1Li
        "6": {"0": 104.9, "45": 64.3, "90": 63.1},  # 6号 Mg-2Zn-3Li
        "7": {"0": 161.0, "45": 171.5, "90": 186.9},  # 7号 AZ31
        "8": {"0": 145.6, "45": 123.6, "90": 128.9},  # 8号 Mg-2Zn-1Gd
    },
    "uts": {  # 0°        45°       90°
        "1": {"0": 361.3, "45": 359.4, "90": 361.2},  # 1号 AZ31 挤压原始态
        "2": {"0": 364.1, "45": 334.3, "90": 343.1},  # 2号 AZ31 8.5%PRC + 退火
        "3": {"0": 324.2, "45": 328.4, "90": 323.8},  # 3号 AZ31 8.5%PRC + 13.5%PRT + 退火
        "4": {"0": 290.8, "45": 293.7, "90": 286.6},  # 4号 Mg-2Zn
        "5": {"0": 264.8, "45": 266.7, "90": 262.1},  # 5号 Mg-2Zn-1Li
        "6": {"0": 265.0, "45": 249.6, "90": 279.8},  # 6号 Mg-2Zn-3Li
        "7": {"0": 337.5, "45": 339.2, "90": 341.2},  # 7号 AZ31
        "8": {"0": 324.1, "45": 318.2, "90": 316.1},  # 8号 Mg-2Zn-1Gd
    },
    "el": {  # 0°        45°       90°
        "1": {"0": 20.3, "45": 19.2, "90": 17.1},  # 1号 AZ31 挤压原始态
        "2": {"0": 22.6, "45": 24.2, "90": 21.9},  # 2号 AZ31 8.5%PRC + 退火
        "3": {"0": 26.7, "45": 24.9, "90": 24.1},  # 3号 AZ31 8.5%PRC + 13.5%PRT + 退火
        "4": {"0": 26.2, "45": 26.8, "90": 23.5},  # 4号 Mg-2Zn
        "5": {"0": 26.9, "45": 32.2, "90": 31.3},  # 5号 Mg-2Zn-1Li
        "6": {"0": 31.0, "45": 37.3, "90": 38.7},  # 6号 Mg-2Zn-3Li
        "7": {"0": 22.4, "45": 24.2, "90": 22.0},  # 7号 AZ31
        "8": {"0": 24.3, "45": 32.0, "90": 21.9},  # 8号 Mg-2Zn-1Gd
    }
}


class Sample():
    def __init__(self, sample_id):
        if int(sample_id) < 1:
            raise Exception("样品id是大于0的整数")
        self.id = str(sample_id)  # 需要字符串作为key

    def prop(self, prop_name, direction=None):
        """
            返回样品的指定拉伸方向上的属性
            :Param direction 拉伸角度
        """
        if prop_name not in prop.keys() or (direction is not None and direction not in prop[prop_name][self.id].keys()):
            raise Exception("属性不存在！（查询条件：%s,%s）" % (prop_name, direction))
        return prop[prop_name][self.id][direction] if direction is not None else prop[prop_name][self.id]

    def ys(self, direction=None):
        """ 屈服强度 """
        return self.prop("ys", direction)

    def uts(self, direction=None):
        """ 抗拉强度 """
        return self.prop("uts", direction)

    def el(self, direction=None):
        """ 延伸率 """
        return self.prop("el", direction)

    def num_of_blocks(self):
        return blocks_num_dict[self.id]


class KFoldLoader():
    def __init__(self, X, Y, k=5):
        self.X = X
        self.Y = Y
        self.k = k
        self.index = 0
        # 留一法
        if k == 1:
            self.k = len(X)  # 留一法指每次只留一个作为验证，其实就是数据分为len(X)组

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.k:
            ret = self.__getdata__()
            self.index += 1
            return ret
        else:
            raise StopIteration  # 通过抛出这个异常，让循环结束

    def __getdata__(self):
        k = self.k
        i = self.index
        X = self.X
        Y = self.Y
        """
            获取k折交叉验证的第 i+1折（i = 0 -> k-1） 训练和验证数据
            (k折验证，将数据划分为k组，每次取一组作为验证，其他为训练。最终每份都会被验证，即交叉训练k次)
        """
        fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）
        val_start = i * fold_size  # 验证集的初始下标
        if i != k - 1:  # 非最后一折
            val_end = (i + 1) * fold_size
            # 验证集
            x_valid = X[val_start:val_end]
            y_valid = Y[val_start:val_end]
            # 训练集（拼接验证集之前和之后的部分）
            x_train = torch.cat((X[0:val_start], X[val_end:]), dim = 0) if isinstance(X,Tensor) else X[0: val_start] + X[val_end:]
            y_train = torch.cat((Y[0:val_start], Y[val_end:]), dim = 0) if isinstance(Y,Tensor) else Y[0: val_start] + Y[val_end:]
        else:  # 若是最后一折交叉验证，可能存在划分数据有剩余
            x_valid, y_valid = X[val_start:], Y[val_start:]  # 若不能整除，将多的case放在最后一折里
            x_train = X[0: val_start]
            y_train = Y[0: val_start]

        return i, x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    # print(Sample(1).prop("el", "90"))
    pass