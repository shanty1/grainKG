import copy
import itertools
import random
import numpy as np
import torch
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
from dgl.data import DGLDataset
import os
import shutil

from torch import Tensor

from CALCULATE_MODULE.GNN import util, repository
from dao import grain_dao
from CALCULATE_MODULE.GNN.data_helper import blocks_num_dict, Sample

"""
    文件附加说明：
        数据库中的节点一定有要2*2，3*3这个标签，用于区分哪些数据是2*2划分的，哪些是3*3划分的。 2*2和3*3的数据是一样的，所以应完全区分他们为单独的数据库
"""

# 图的类别选择，选择加在数据库异构图或同构图
_HETERO, _HOMO = 'ebsd-heterograph', 'ebsd-homograph'


def validate_dataset_size(graph_database):
    matrix_size_str = graph_database.replace("DB", "")
    m, n = matrix_size_str.split("*")
    if not (graph_database.startswith("DB") and m.isdigit() and int(m) > 0 and n == m):
        raise Exception("晶粒数据库标签格式错误！graph_database的格式只能是2*2, 3*3, ...，以此类推。 错误标签格式： " + graph_database)
    # 获取graph_database  数据库中的总子图数量
    num_of_subgraphs = grain_dao.count_all_labels_of_grains(graph_database)
    # 预设的每个样品块（Sample:Block）划分子图（Sub）的数量
    num_of_sub = eval(matrix_size_str)
    # 判断{graph_database}数据库中的子图数量与程序里配置的子图数量（blocks_num_dict中配置的块数量*每块划分子图的数量）是否不一致
    if not num_of_subgraphs == sum(blocks_num_dict.values()) * num_of_sub:
        raise Exception("\n数据集划分异常！%s中配置的数据数量与数据库中的数据量不一致，请检查！\n"
                        "配置数量: %d,  (%s)数据库数量: %d" % (str(blocks_num_dict),
                                                      sum(blocks_num_dict.values()) * num_of_sub,
                                                      graph_database, num_of_subgraphs))
    return num_of_sub


def split_dataset_for_sub_predict_task(graph_type=_HETERO, graph_database="DB3*3", test_size=0.2, label_options="ys", directions="90",
                                       scaler_x=None, update_attr_nodes=False, self_loop=True,
                                       random_state=None):
    """
        读取数据库并划分数据集
        test_size 测试集个数或者比例
    """
    setting_of_num_of_sub = validate_dataset_size(graph_database)
    train_graphs, train_labels, test_graphs, test_labels = [], [], [], []
    if isinstance(label_options, str): label_options = [label_options]
    if isinstance(directions, str): directions = [directions]
    print("测试集大小为%.2f划分结果：（1作为训练集，0作为测试集）" % test_size)
    random.seed(random_state)
    for sample_id, block_num in blocks_num_dict.items():  # 逐个样本
        print("样品%s的" % sample_id)
        for block_id in range(1, block_num + 1):  # 逐个遍历每个样本的block
            sub_of_block_test_size = test_size if isinstance(test_size, int) else int(setting_of_num_of_sub * test_size)     # 测试集数量
            test_sub_ids = random.sample(list(range(1, setting_of_num_of_sub + 1)), sub_of_block_test_size)  # 随机抽取测试子图的编号
            for sub_id in range(1, setting_of_num_of_sub + 1):
                print("\t %d" % sub_id,
                      "测试集" if sub_id in test_sub_ids else "训练集",
                      end=": ")
                ''' ## 这里的label非常重要！！  有五个标签一个不能少某个'''
                this_type_node_lables = [graph_database, "Grain",
                                         "Sample%s" % sample_id, "Block%d" % block_id, "Sub%d" % sub_id]
                if graph_type == _HETERO:
                    graph = repository.heterograph_feature(this_type_node_lables, update_attr_nodes, self_loop, scaler_x)  # scaler暂时没用
                elif graph_type == _HOMO:
                    graph = repository.homograph_feature(this_type_node_lables, scaler_x)  # scaler暂时没用

                if sub_id in test_sub_ids:  # 如果子图id==子图测试集id
                    test_graphs.append(graph)
                    test_labels.append([Sample(sample_id).prop(l, d) for l,d in itertools.product(label_options,directions)])
                else:
                    train_graphs.append(graph)
                    train_labels.append([Sample(sample_id).prop(l, d) for l, d in itertools.product(label_options, directions)])
    print("数据集构建完毕，训练集：%d条，测试集：%d条。" % (len(train_labels), len(test_labels)))
    return train_graphs, train_labels, test_graphs, test_labels


def split_dataset_for_block_predict_task(graph_type=_HETERO, graph_database="DB3*3", test_size=0.2, label_options="ys", directions="90",
                                       scaler_x=None, update_attr_nodes=False, self_loop=True,
                                       random_state=None):
    """
        读取数据库并划分数据集
        test_size 测试集个数或者比例
    """
    setting_of_num_of_sub = validate_dataset_size(graph_database)
    train_graphs, train_labels, test_graphs, test_labels = [], [], [], []
    if isinstance(label_options, str): label_options = [label_options]
    if isinstance(directions, str): directions = [directions]
    print("测试集大小为%.2f划分结果：（1作为训练集，0作为测试集）" % test_size)
    random.seed(random_state)
    # train_mask = np.array([], dtype=bool)
    for sample_id, block_num in blocks_num_dict.items():  # 逐个样本
        print("样品%s的切块：" % sample_id)
        num_of_testset = test_size if isinstance(test_size, int) else int(block_num*test_size)  # 每个样品的测试block数量
        test_block_ids = random.sample(list(range(1, block_num+1)), num_of_testset)   # 随机抽样测试集的 Block ID
        # 开始遍历block，如果block为测试，所有子图都为测试
        for block_id in range(1, block_num + 1):  # 逐个遍历每个样本的block , 注意数据类型都得是int
            print("\t %d" % block_id,
                  "测试集" if block_id in test_block_ids else "训练集",
                  end="(%d个子图) \n" % setting_of_num_of_sub)
            # 、 遍历block的子图们
            for sub_id in range(1, setting_of_num_of_sub + 1):
                ''' ## 这里的label非常重要！！  有五个标签一个不能少某个'''
                this_type_node_lables = [graph_database, "Grain",
                                         "Sample%s" % sample_id, "Block%d" % block_id, "Sub%d" % sub_id]
                print("\t\t", end="")
                if graph_type == _HETERO:
                    graph = repository.heterograph_feature(this_type_node_lables, update_attr_nodes, self_loop, scaler_x)  # scaler暂时没用
                elif graph_type == _HOMO:
                    graph = repository.homograph_feature(this_type_node_lables, scaler_x)  # scaler暂时没用

                if block_id in test_block_ids:  # 如果子图id==子图测试集id
                    test_graphs.append(graph)
                    # itertools.combinations((directions,label_options) , 2)
                    test_labels.append([Sample(sample_id).prop(l, d) for l,d in itertools.product(label_options,directions)])
                else:
                    train_graphs.append(graph)
                    train_labels.append([Sample(sample_id).prop(l, d) for l, d in itertools.product(label_options, directions)])
    print("数据集构建完毕，训练集：%d条，测试集：%d条。" % (len(train_labels), len(test_labels)))
    return train_graphs, train_labels, test_graphs, test_labels


def split_dataset_for_sample_predict_task(graph_type, graph_database, test_size, label_options, directions,
                                          update_attr_nodes, self_loop, scaler_x, random_state):
    setting_of_num_of_sub = validate_dataset_size(graph_database)
    train_graphs, train_labels, test_graphs, test_labels = [], [], [], []
    if isinstance(label_options, str): label_options = [label_options]
    if isinstance(directions, str): directions = [directions]
    print("测试集大小为%.2f划分结果：（1作为训练集，0作为测试集）" % test_size)
    random.seed(random_state)

    sample_num = len(blocks_num_dict.keys())
    num_of_testset = test_size if isinstance(test_size, int) else int(sample_num * test_size)  # 每个样品的测试block数量
    test_sample_ids = random.sample(list(range(1, sample_num + 1)), num_of_testset)  # 随机抽样测试集的 Block ID
    for sample_id, block_num in blocks_num_dict.items():  # 逐个样本
        sample_id = int(sample_id)
        print("样品%s  %s：" % (sample_id,"测试集" if sample_id in test_sample_ids else "训练集"))
        # 开始遍历block，如果block为测试，所有子图都为测试
        for block_id in range(1, block_num + 1):  # 逐个遍历每个样本的block , 注意数据类型都得是int
            print("\t 切块%d" % block_id, "(%d个子图) " % setting_of_num_of_sub)
            # 、 遍历block的子图们
            for sub_id in range(1, setting_of_num_of_sub + 1):
                ''' ## 这里的label非常重要！！  有五个标签一个不能少某个'''
                this_type_node_lables = [graph_database, "Grain",
                                         "Sample%s" % sample_id, "Block%d" % block_id, "Sub%d" % sub_id]
                print("\t\t", end="")
                if graph_type == _HETERO:
                    graph = repository.heterograph_feature(this_type_node_lables, update_attr_nodes, self_loop,
                                                           scaler_x)  # scaler暂时没用
                elif graph_type == _HOMO:
                    graph = repository.homograph_feature(this_type_node_lables, scaler_x)  # scaler暂时没用

                if sample_id in test_sample_ids:  # 如果子图id==子图测试集id
                    test_graphs.append(graph)
                    # itertools.combinations((directions,label_options) , 2)
                    test_labels.append(
                        [Sample(sample_id).prop(l, d) for l, d in itertools.product(label_options, directions)])
                else:
                    train_graphs.append(graph)
                    train_labels.append(
                        [Sample(sample_id).prop(l, d) for l, d in itertools.product(label_options, directions)])
    print("数据集构建完毕，训练集：%d条，测试集：%d条。" % (len(train_labels), len(test_labels)))
    return train_graphs, train_labels, test_graphs, test_labels


def split_dataset(graph_type, graph_database, test_size, target, label_options, directions, update_attr_nodes,
                  self_loop, scaler_x, random_state):
    """
        根据不同目标任务划分数据集
        target 预测目标，必须是["sub", "block", "sample"]其中之一
        注:  sub指在子图上进行数据集划分（测试子图才是测试集）
            block指在样本块上进行数据集划分（测试块下的子图都是测试集）
            sample指在样本上进行数据集划分（测试样本下的子块的子图都是测试集）
            (最终的输入都是子图，只是划分的依据层级如上不同)
    """
    if target == "sub":
        return split_dataset_for_sub_predict_task(graph_type=graph_type, graph_database=graph_database, test_size=test_size,
                      label_options=label_options, directions=directions,
                      update_attr_nodes=update_attr_nodes, self_loop=self_loop,
                      scaler_x=scaler_x, random_state=random_state)
    elif target == "block":
        return split_dataset_for_block_predict_task(graph_type=graph_type, graph_database=graph_database, test_size=test_size,
                                           label_options=label_options, directions=directions,
                                           update_attr_nodes=update_attr_nodes, self_loop=self_loop,
                                           scaler_x=scaler_x, random_state=random_state)
    elif target == "sample":
        return split_dataset_for_sample_predict_task(graph_type=graph_type, graph_database=graph_database, test_size=test_size,
                                           label_options=label_options, directions=directions,
                                           update_attr_nodes=update_attr_nodes, self_loop=self_loop,
                                           scaler_x=scaler_x, random_state=random_state)
    else:
        raise Exception("不支持此类目标任务！")


def get_dataset(graph_type, graph_database, test_size=0.2, target="block",
                label_options="ys", directions="90", force_reload = False,
                self_loop=True, scaler_x=None, scaler_y=None, verbose=True,
                update_attr_nodes=False, random_state=None):
    # 序列化存储名字标识
    params =  str(graph_type) + str(graph_database) + str(test_size)      \
            + str(target)     + str(label_options)  + str(directions)     \
            + ('self_loop' if self_loop else 'no_self_loop')               \
            + ('upd_attr'  if update_attr_nodes else 'no_upd_attr')        \
            + ('scaler_x'  if scaler_x else 'no_scaler_x')                 \
            + ('scaler_y'  if scaler_y else 'no_scaler_y')
    save_dataloader_filename_train = util.md5(params + "train") + ".bin"
    save_dataloader_filename_test  = util.md5(params + "test") + ".bin"
    # 打印信息
    print("----------------加载[%s]数据集----------------" % graph_type
          + " \n选择图切割方式: %s" % graph_database
          + " \n测试集划分占比: %.2f" % test_size
          + "      \n预测目标: %s" % target
          + "   \n选取性能标签: %s" % str(label_options)
          + "   \n选取性能方向: %s" % str(directions)
          + "\n是否添加边自循环: %s" % ("是" if self_loop else "否")
          + "\n是否更新属性节点: %s" % ("是" if update_attr_nodes else "否")
          + "\n是否图特征标准化: %s" % ("是" if scaler_x  else "否")
          + "\n是否标签值标准化: %s" % ("是" if scaler_y  else "否"), end="\n"*2)

    train_graphs, train_labels, test_graphs, test_labels = None, None, None, None
    if force_reload:
        if graph_type in [_HETERO, _HOMO]:
            train_graphs, train_labels, test_graphs, test_labels \
                = split_dataset(graph_type=graph_type, graph_database=graph_database, test_size=test_size,
                                                     target=target, label_options=label_options, directions=directions,
                                                     update_attr_nodes=update_attr_nodes, self_loop=self_loop,
                                                     scaler_x=scaler_x, random_state=random_state)
            # 归一化
            # 注意：测试集的归一化和逆转应该使用训练集的均值和方差！！！！
            if scaler_y:
                scaler_y.fit(np.array(train_labels))
                train_labels = scaler_y.transform(np.array(train_labels)).astype(np.float32)
                test_labels = scaler_y.transform(np.array(test_labels)).astype(np.float32)
        else:
            raise Exception("请从%s选择图类型" % str([_HETERO, _HOMO]) )
    else: # 缓存加载，scaler_y同样也从缓存中读取
        scaler_y = None

    # 构造dataloader（scaler_y必须已经fit）
    train_dataset = EbsdDataset(train_graphs, train_labels,scaler_y,
                                force_reload=force_reload, save_filename=save_dataloader_filename_train, verbose=verbose)
    test_dataset = EbsdDataset(test_graphs, test_labels,scaler_y,
                               force_reload=force_reload, save_filename=save_dataloader_filename_test, verbose=verbose)

    return train_dataset,test_dataset


class EbsdDataset(DGLDataset):
    def __init__(self, graphs, labels, fited_scaler_y=None,
                 save_filename=None, force_reload=False,verbose=False):

        # 初始化成员变量
        self.graphs = graphs  # 数据集-特征图
        self.labels = labels  # 数据集-标签
        self.save_filename = save_filename # 数据存储完整名字

        # 用于inverse，不用于transform。
        # 无需使用，此处只是为了同数据一起序列化，缓存加载的时候可能要用到inverse，而防止原来fit后的scaler丢失
        # 成员labels 一定是经过 成员fited_scaler_y转化来的
        self.fited_scaler_y = fited_scaler_y

        # ！！！构造方法
        super(EbsdDataset, self).__init__(name="EbsdDataset", force_reload=force_reload, verbose=verbose)

    def process(self):
        if self.graphs == None or len(self.graphs) == 0:
            raise Exception("数据为空！请使用强制重新加载，或检查数据库！")
        elif len(self.labels) != len(self.graphs) :
            raise Exception("标签数量不匹配！")

        self.graphs = copy.deepcopy(self.graphs)
        self.labels = copy.deepcopy(self.labels)

        if not isinstance(self.labels, Tensor):  # only tensor can be serialized
            self.labels = torch.tensor(self.labels)

    def download(self):
        pass

    def save(self):
        if self.save_filename is None:
            if self.verbose: print("保存文件名为空，不进行数据集序列化~")
            return None
        try:
            graph_path = os.path.join(self.save_path, self.save_filename)
            save_graphs(graph_path, self.graphs, {'labels': self.labels})
            # 在Python字典里保存其他信息
            info_path = os.path.join(self.save_path, self.save_filename + '_info.pkl')
            save_info(info_path, {'fited_scaler_y': self.fited_scaler_y})

            if self.verbose: print("saved to path: ", graph_path)
        except Exception as e:
            print("Error：", e)
            shutil.rmtree(self.save_path)

    def load(self):
        graph_path = os.path.join(self.save_path, self.save_filename)
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

        info_path = os.path.join(self.save_path, self.save_filename + '_info.pkl')
        info = load_info(info_path)
        self.fited_scaler_y = info['fited_scaler_y']

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.save_filename)
        return os.path.exists(graph_path)

    def __getitem__(self, idx):
        """ 通过idx获取对应的图和标签 """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """数据集中图的数量"""
        return len(self.graphs)

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return self.labels.shape[-1]


if __name__ == '__main__':
    # split_mask_for_sub_predict_task("DB2*2")
    split_dataset_for_sub_predict_task("DB2*2")
