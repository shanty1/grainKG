import dgl
import dgl.nn as dglnn
import torch
from dgl._deprecate.graph import DGLGraph
from dgl.nn.pytorch import GATConv
from torch import nn
import torch.nn.functional as F

import config

def sum_with_mutilheads(tensors, dsttype):
    """
        子图聚合函数
    """
    # tensors: is a list of tensors to aggregate
    # dsttype: string name of the destination node type for which the
    #          aggregation is performed
    stacked = torch.stack(tensors, dim=0)
    h = torch.sum(stacked, dim=0)
    h = h.mean(dim=1) # 多头
    return h

class HGAT_DNN(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_heads,
                 feat_drop=0, attn_drop=0, nn_drop=0, negative_slope=0.2, activation=F.elu, residual=False,
                 num_cov_layer=3, num_nn_layer=3, readout=False):
        print("----------------构建模型----------------"
              + "\n     卷积网络层数: %d" % num_cov_layer
              + "\n  子网络注意力头数: %d" % num_heads
              + "\n  子网络使用残差？: %s" % residual
              + "\n     映射网络层数: %d" % num_nn_layer
              + "\n     映射输入方式: %s" % "readout" if readout else "all nodes"
              + "\n 卷积输入/输出大小: 源/目标 节点大小"
              + "\n映射网络隐藏层大小: %d" % hidden_dim
              + "\n       输出层大小: %d" % out_dim , end="\n"*2)

        self.num_cov_layer = num_cov_layer
        self.num_nn_layer = num_nn_layer
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.nn_drop = nn_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.activation = activation
        self.residual = residual
        self.readout = readout
        super(HGAT_DNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_cov_layer):
            self.conv_layers.append(
                dglnn.HeteroGraphConv({                # in_dim=(源节点特征大小，目标节点特征大小)
                                                   # hidden_dim=输出特征大小  （注：最终的输出特征大小需要与目标节点大小一致）
                    'StrongBoundWith':  self.gat_conv(
                                                        in_dim = (config.grain_node_size, config.grain_node_size),
                                                    hidden_dim = config.grain_node_size),
                    'WeakBoundWith':    self.gat_conv(
                                                        in_dim = (config.grain_node_size, config.grain_node_size),
                                                    hidden_dim = config.grain_node_size),
                    'eular':            self.gat_conv(
                                                        in_dim = (config.grain_node_size, config.per_eular_class_num ** 3),
                                                    hidden_dim = config.per_eular_class_num ** 3),
                    'size':             self.gat_conv(
                                                        in_dim = (config.grain_node_size, config.grainsize_node_class_num),
                                                    hidden_dim = config.grainsize_node_class_num),
                    'eular_':           self.gat_conv(
                                                        in_dim = (config.per_eular_class_num ** 3, config.grain_node_size),
                                                    hidden_dim = config.grain_node_size),
                    'size_':            self.gat_conv(
                                                        in_dim = (config.grainsize_node_class_num, config.grain_node_size),
                                                    hidden_dim = config.grain_node_size)
                    }, aggregate=sum_with_mutilheads)
            )
        module_list = []
        input_dim = config.grain_node_size
        for i in range(num_nn_layer):
            module_list.append(nn.Linear(input_dim, hidden_dim))
            # module_list.append(nn.BatchNorm1d(hidden_dim))  # 测试表明收敛速度加快，但是很大程度欠拟合
            module_list.append(nn.ReLU())
            input_dim = hidden_dim
            module_list.append(nn.Dropout(p=self.nn_drop))
        module_list.append(nn.Linear(hidden_dim, out_dim))
        self.linears = nn.Sequential(*module_list)

    ''' HeteroGraphConv图默认将allow_zero_in_degree转为True '''
    def gat_conv(self, in_dim, hidden_dim):
        return GATConv(in_dim, hidden_dim, num_heads=self.num_heads,
                                      feat_drop=self.feat_drop,attn_drop=self.attn_drop,negative_slope=self.negative_slope,
                                      activation=self.activation,residual=self.residual,allow_zero_in_degree=True)

    def forward(self, g, return_feat=False):
        with g.local_scope():
            h = g.ndata['h'] # 不需要分收尾节点特征，因为对于同一类节点，不管它是首还是尾，都是一样的
            for i in range(self.num_cov_layer):
                h = self.conv_layers[i](g, h)
            # 最终节点特征赋予图上
            g.ndata['h'] = h

            if return_feat:
                if g.batch_size>1:
                    graphs = dgl.unbatch(g)
                    return torch.stack([graph.ndata['h']['grain'] for graph in graphs],dim=0)
                else:
                    return torch.stack([g.ndata['h']['grain']])

            if self.readout: # 走readout模式
                mean_of_nodes_feature = dgl.mean_nodes(g, 'h', ntype='grain') # batch*node_feature
                out = self.linears(mean_of_nodes_feature)

            else: # 走allnode模式（将每个node特征作为图特征，进行非线性转换，然后求平均）
                graphs = dgl.unbatch(g)
                out = [self.linears(graph.ndata['h']['grain']).mean(dim=0).unsqueeze(0) for graph in graphs]
                out = torch.cat(out, dim=0) # 使用cat将list拼接成tensor，不会失去梯度
            return out



class HomoGAT(nn.Module):
    def __init__(self,
                 in_dim,  hidden_dim, out_dim, heads,
                 feat_drop=0, attn_drop=0, nn_drop=0, negative_slope=0.2, activation=F.elu, residual=False,
                 num_cov_layer=3, num_nn_layer=3, readout=False):
        super(HomoGAT, self).__init__()
        self.num_cov_layer = num_cov_layer
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.readout = readout
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, hidden_dim, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_cov_layer-1):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.gat_layers.append(GATConv(
                hidden_dim * heads[l-1], hidden_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            hidden_dim * heads[-2], hidden_dim, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        # linear layer
        module_list = []
        for i in range(num_nn_layer-1):
            module_list.append(nn.Linear(hidden_dim, hidden_dim))
            # module_list.append(nn.BatchNorm1d(hidden_dim))  # 测试表明收敛速度加快，但是很大程度欠拟合
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(p=nn_drop))
        module_list.append(nn.Linear(hidden_dim, out_dim))
        self.linears = nn.Sequential(*module_list)

    def forward(self, g, return_feat=False):
        with g.local_scope():
            h = g.ndata['h'] # 不需要分收尾节点特征，因为对于同一类节点，不管它是首还是尾，都是一样的
            for i in range(self.num_cov_layer):
                h = self.gat_layers[i](g, h).flatten(1)  # 多个头碾平
            # 最终节点特征赋予图上
            g.ndata['h'] = h

            if return_feat:
                if g.batch_size>1:
                    graphs = dgl.unbatch(g)
                    return torch.stack([graph.ndata['h']for graph in graphs],dim=0)
                else:
                    return torch.stack([g.ndata['h']])

            if self.readout: # 走readout模式
                mean_of_nodes_feature = dgl.mean_nodes(g, 'h') # batch*node_feature
                out = self.linears(mean_of_nodes_feature)

            else: # 走allnode模式（将每个node特征作为图特征，进行非线性转换，然后求平均）
                graphs = dgl.unbatch(g)
                out = [self.linears(graph.ndata['h']).mean(dim=0).unsqueeze(0) for graph in graphs]
                out = torch.cat(out, dim=0) # 使用cat将list拼接成tensor，不会失去梯度
            return out