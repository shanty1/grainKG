import copy

import py2neo
import dgl
import torch
import pandas as pd
import random
import pandas as pd
import sklearn as  skl
import numpy as np

from CALCULATE_MODULE.GNN import config, util
from dao import grain_dao

'''
   生成节点标签条件
'''
def generate_labels_c(labels):
    labels_copy = copy.deepcopy(labels) if isinstance(labels,list) else copy.deepcopy(labels).tolist()
    labels_copy.reverse() # 保持打印的顺序
    labels_c = ''
    while len(labels_copy) > 0:
        labels_c += ':`' + labels_copy.pop() + '`'
    print("labels-->",labels_c)
    return labels_c

'''
    根据节点标签条件查询所有节点、节点间关系
'''
def read_graph_from_neo4j(labels, node_name="id", source_name="u_id", target_name="v_id"):
    labels_c = generate_labels_c(labels)
    cypher_query_nodes = "MATCH (n{}) RETURN id(n) as {}, labels(n) as labels, properties(n) as properties order by id asc".format(labels_c,node_name)
    cypher_query_edges = "MATCH (u{})-[r]->(v{}) RETURN id(u) as {}, id(v) as {}, id(r) as r_id, type(r) as r_type, properties(r) as properties order by r_id asc".format(labels_c,labels_c,source_name,target_name)

    # 连接数据库
    graph = grain_dao.graph
    # 查询，并使用.data()序列化数据
    nodes_data = graph.run(cypher_query_nodes).data()
    links_data = graph.run(cypher_query_edges).data()
    # print("nodes_num-->", len(nodes_data))
    # print("edges_num-->", len(links_data))
    if len(nodes_data) == 0:
        raise Exception("查询节点{}数量为0！".format(labels_c))
    return nodes_data,links_data

"""
    构建同构图
    根据`标签`从数据库读取图数据，并创建dgl.graph和feature，只读neo4j图中的关系。即近晶粒相邻关系
    feature_props：预设的feature构成
    node{
        'feat':[`angle`,`boundLength`,`ellipticity`,`phi`,`phi1`,`phi2`,`size`,`surface`]
    }
    edge{
        'bl': `boundLength`
        'dis': `disorientation`
    }
"""
# feature_props = ['angle','boundLength','ellipticity','phi','phi1','phi2','size','surface']
feature_props = ['phi','phi1','phi2','size']
def homograph_feature(labels,  scaler_x=None):
    nodes_data, links_data = read_graph_from_neo4j(labels)
    nodes = [n['id'] for n in nodes_data]
    # 对结点重新编号，创建字典映射
    node_dic = pd.Series(data=range(len(nodes)), index=nodes)
    # 创建关联边，编号根据字典映射
    u, v = [node_dic[n['u_id']] for n in links_data], [ node_dic[n['v_id']] for n in links_data]
    # 取出边的晶界长度、晶粒间的取向差
    bl, dis = [n['properties']['boundLength'] for n in links_data], [n['properties']['disorientation'] for n in links_data]
    graph = dgl.graph((u,v))
    # 创建特征向量
    feature = torch.tensor([])
    for prop in feature_props:
        prop_feat = torch.tensor([[n['properties'][prop]] for n in nodes_data]) #
        feature = torch.cat([feature,prop_feat],dim=1)
    if scaler_x:  # 这里还没有区分测试集，使用归一化的话只在本张图内做了归一化，所以不建议使用
        feature = torch.from_numpy(scaler_x.fit_transform(feature)).float()
        bl = scaler_x.fit_transform(torch.tensor(bl).unsqueeze(1)).flatten()
        dis = scaler_x.fit_transform(torch.tensor(dis).unsqueeze(1)).squeeze(1)
    # 设置边的属性
    graph.edata['bl'] = torch.tensor(bl).float()  # 边界长度
    graph.edata['dis'] = torch.tensor(dis).float()  # 取向差
    # 图生成结点是根据边来的，如果有孤立结点则不会加到图中，需要手动添加结点，或者删除特征数量，否则会出现不匹配报错
    # 选择删除节点特征，孤立结点对图没有意义，少的结点 或多出来的特征必定是比当前图里的结点大的(因为图会自动填充中间结点)，
    # 这里选择添加结点，因为中间可能还有孤立的结点自动填充了，所以上面操作无法杜绝孤立结点
    # 所以从后往前加结点/删特征就行
    graph.add_nodes(len(feature) - graph.number_of_nodes())
    graph.ndata['h'] = feature
    # print("feature compose-->",feature_props)
    # print("feature size-->",feature.shape)
    return graph

"""
    获取晶粒的结点特征嵌入向量
    结点四个属性的值嵌入向量拼接
"""
def get_grain_node_feat_embedding():
    return torch.zeros(config.grain_node_size)


"""
    按顺序生成欧拉角字典,根据配置文件的最大角，最小角和划分数量
    axi_to_id
    id_to_onehot
"""
def generate_eular_dic():
    class_lables = []
    order = 0
    axi_to_id = {}
    for i in range(config.per_eular_class_num):
        for j in range(config.per_eular_class_num):
            for k in range(config.per_eular_class_num):
                axi = str(i)+"_"+str(j)+"_"+str(k)
                class_lables.append(axi)
                axi_to_id[axi] = order
                order += 1
    axi_to_onehot = pd.get_dummies(class_lables)
    # print(axi_to_onehot)
    return axi_to_id, axi_to_onehot.values

"""
    将所有欧拉角onehot, 注意此方法与generate_eular_dic()的字典不匹配,这里根据实际数据生成的可能不包含某些区间类别，而generate_eular_dic会生成所有预置区间类别
    @param eulars: 欧拉角列表
"""
def get_eular_node_feat_onehot_embedding(eulars):
    phi1_step = (config.grain_phi1_max - config.grain_phi1_min)/config.per_eular_class_num
    phi_step  = (config.grain_phi_max  -  config.grain_phi_min)/config.per_eular_class_num
    phi2_step = (config.grain_phi2_max - config.grain_phi2_min)/config.per_eular_class_num
    eulars_axis = []
    for eular in eulars:
        phi1, phi, phi2 = eular
        axi = '{}_{}_{}'.format(
            util.get_num_space_idx(phi1, [(config.grain_phi1_min + phi1_step*i, config.grain_phi1_min + phi1_step*(i+1)) for i in range(config.per_eular_class_num)]),
            util.get_num_space_idx(phi,  [(config.grain_phi_min + phi_step*i, config.grain_phi_min + phi_step*(i+1)) for i in range(config.per_eular_class_num)]),
            util.get_num_space_idx(phi2, [(config.grain_phi2_min + phi2_step*i, config.grain_phi2_min + phi2_step*(i+1)) for i in range(config.per_eular_class_num)]))
        eulars_axis.append(axi)
    axi_to_onehot = pd.get_dummies(eulars_axis)
    return axi_to_onehot.values

def get_eular_node_feat_classno(eular,axi_to_idx):
    phi1_step = (config.grain_phi1_max - config.grain_phi1_min)/config.per_eular_class_num
    phi_step  = (config.grain_phi_max  -  config.grain_phi_min)/config.per_eular_class_num
    phi2_step = (config.grain_phi2_max - config.grain_phi2_min)/config.per_eular_class_num
    (phi1, phi, phi2) = eular
    axi = '{}_{}_{}'.format(
        util.get_num_space_idx(phi1, [(config.grain_phi1_min + phi1_step*i, config.grain_phi1_min + phi1_step*(i+1)) for i in range(config.per_eular_class_num)]),
        util.get_num_space_idx(phi,  [(config.grain_phi_min + phi_step*i, config.grain_phi_min + phi_step*(i+1)) for i in range(config.per_eular_class_num)]),
        util.get_num_space_idx(phi2, [(config.grain_phi2_min + phi2_step*i, config.grain_phi2_min + phi2_step*(i+1)) for i in range(config.per_eular_class_num)]))
    return axi_to_idx[axi]

"""
    获取晶粒度<暂使用晶粒大小代替>结点嵌入-onehot
"""
def get_grainsize_node_feat_onehot_embedding(sizes):
    step = config.grain_size_step
    start = config.grain_size_min
    sizes_classes = []
    for size in sizes:
        size_class = util.get_num_space_idx(size, [(start + step*i, start + step*(i+1)) for i in range(config.grainsize_node_class_num)]),
        sizes_classes.append(str(size_class))
    size_to_onehot = pd.get_dummies(sizes_classes)
    return size_to_onehot.values

def get_grainsize_node_feat_classno(size):
    step = config.grain_size_step
    start = config.grain_size_min
    sizelevel = util.get_num_space_idx(size, [(start + step*i, start + step*(i+1)) for i in range(config.grainsize_node_class_num)]),
    return sizelevel[0]

"""
    建立异构图
"""
def heterograph_feature(labels, update_attr_nodes=False,self_loop=True, scaler_x=None ):
    nodes_data, links_data = read_graph_from_neo4j(labels)
    nodes = [n['id'] for n in nodes_data]
    # 对结点重新编号，创建字典映射
    node_dic = pd.Series(data=range(len(nodes)), index=nodes)
    heteroG = {}
    # 1强相关图 (此处注意构图时，赋边采用元祖和列表的区别)
    stro_r_graph_edge = [[node_dic[n['u_id']], node_dic[n['v_id']]] for n in links_data if n['r_type']=='StrongBoundRelation']
    stro_r = ('grain', 'StrongBoundWith', 'grain')
    heteroG[stro_r] = stro_r_graph_edge
    # 2弱相关图 (此处注意构图时，赋边采用元祖和列表的区别)
    weak_r_graph_u = [node_dic[n['u_id']]for n in links_data if n['r_type']=='WeakBoundRelation']
    weak_r_graph_v = [node_dic[n['v_id']]for n in links_data if n['r_type']=='WeakBoundRelation']
    weak_r = ('grain', 'WeakBoundWith', 'grain')
    heteroG[weak_r] = (weak_r_graph_u,weak_r_graph_v)
    # 3欧拉角节点图(无向图)
    eular_r_graph_u = [node_dic[n['id']] for n in nodes_data ]
    axi_to_idx, axi_to_onehot = generate_eular_dic()
    eular_r_graph_v = [
        get_eular_node_feat_classno(
            (n['properties']['phi1'],
             n['properties']['phi'],
             n['properties']['phi2']),
             axi_to_idx
        )for n in nodes_data]
    eular_r = ('grain', 'eular', 'eularlevel')
    if update_attr_nodes:
        heteroG[eular_r] = (eular_r_graph_u, eular_r_graph_v)
    # 无向图
    eular_r_ = ('eularlevel', 'eular_', 'grain')
    heteroG[eular_r_] = (eular_r_graph_v, eular_r_graph_u)
    eularlevel_feat = axi_to_onehot[:np.max(eular_r_graph_v)+1]
    # 3晶粒大小节点图(无向图)
    size_r_graph_u = [node_dic[n['id']] for n in nodes_data ]
    size_r_graph_v = [get_grainsize_node_feat_classno(n['properties']['size']) for n in nodes_data]
    size_r = ('grain', 'size', 'sizelevel')
    if update_attr_nodes:
        heteroG[size_r] = (size_r_graph_u, size_r_graph_v)
    # 无向图
    size_r_ = ('sizelevel', 'size_', 'grain')
    heteroG[size_r_] = (size_r_graph_v, size_r_graph_u)
    sizelevel_feat = (pd.get_dummies([i for i in range(config.grainsize_node_class_num)]).values)[:np.max(size_r_graph_v)+1]
    # print(sizelevel_feat)
    # 创建特征向量
    # graph.ndata['feat'] = np.expand_dims(get_grain_node_feat_embedding(),0).repeat(graph.number_of_nodes(),axis=0)
    g = dgl.heterograph(heteroG)
    g.nodes['grain'].data['h'] = torch.from_numpy(np.expand_dims(get_grain_node_feat_embedding(),0).repeat(len(nodes), axis=0)).float()
    g.nodes['eularlevel'].data['h'] = torch.from_numpy(eularlevel_feat).float()
    g.nodes['sizelevel'].data['h'] = torch.from_numpy(sizelevel_feat).float()
    if self_loop:
        g = dgl.add_self_loop(g,('grain', 'StrongBoundWith', 'grain'))
        g = dgl.add_self_loop(g,('grain', 'WeakBoundWith', 'grain'))
    return g


"""
    获取数据库中所有的图
    标签名称需要管理员与数据库匹配
"""
labels = ':`No.1`:`Grain`:`Sample`'
def batch_graph_feature():
    pass


if __name__=='__main__':
    # 连接数据库
    graph = py2neo.Graph(config.uri, username=config.username, password=config.password)
    # 查询，并使用.data()序列化数据
    result = graph.run("call db.labels").data()
    print(result)