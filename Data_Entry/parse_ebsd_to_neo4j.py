import _thread
import os
import warnings
import pandas as pd
import numpy as np
import py2neo
from py2neo import Node, Relationship
from dao import grain_dao
from resources import conf

# 数据文件名
file_path = "./Grain_Files/";
grain_filename = "ATEX_OUT_InfosGrains.csv";
pixels_filename = "ATEX_OUT_InfosGrainsPixels.csv";
relation_filename = "ATEX_OUT_Neighbors.csv";

# 晶粒图的切割方法，用于标识晶粒属于哪种切割，每一种切割命名为一种数据库。晶粒数据库名，使用一个数据库，用标签来区分数据库
grain_dbs = ("DB1*1", "DB2*2", "DB3*3");
# 晶粒边划分阈值
threshold_of_grain_boundary_length = 0.3

# 数据库操作接口
global_graph =  py2neo.Graph("bolt://192.168.31.26:7687/", auth=("neo4j", "123456"))

def generate_sub_label(grain_axis, max_axis, divide_num):
    """
        根据晶粒坐标、坐标系和切割数量来计算晶粒所属区间，生成子图标签。
    """
    num_of_block_per_axis = round(divide_num ** 0.5)  # 每个坐标轴划分区间数量
    step_width  = max_axis[0] / num_of_block_per_axis # x轴的区间大小
    step_height = max_axis[1] / num_of_block_per_axis # y轴的区间大小
    w, h = None, None # 晶粒处于x和y方向上的第几区间
    for j in range(num_of_block_per_axis):
        if grain_axis[0] >= j*step_width  and grain_axis[0] <= (j+1)*step_width:
            w = j
        if grain_axis[1] >= j*step_height and grain_axis[1] <= (j+1)*step_height:
            h = j
    if w is None or h is None:
        raise Exception("晶粒划分子图失败！")
    return "Sub%d" % (h * num_of_block_per_axis + w + 1)

def eval_and_return_db_dividesize(grain_db):
    """
        验证晶粒切割设置是否正确，并返回切割数量。切割标识必须同 #grain_dbs
    """
    matrix_size_str = grain_db.replace("DB", "")
    m, n = matrix_size_str.split("*")
    if not (grain_db.startswith("DB") and m.isdigit() and int(m)>0 and n == m):
        raise Exception("晶粒数据库标签格式错误！ 错误标签格式： " + grain_db)
    return eval(matrix_size_str)


def generate_bound_relation(node_u, node_v, common_bound_length, disorientation):
    """
        生成图谱晶粒间的晶界关系
    """
    u_bound_length = node_u['boundLength']
    u_bound_length_ratio = common_bound_length / u_bound_length
    r_type = "WeakBoundRelation" if u_bound_length_ratio <= threshold_of_grain_boundary_length else "StrongBoundRelation"
    return Relationship(node_u, r_type, node_v, boundLength=common_bound_length, disorientation=disorientation)

def parse_file_to_graph(graph, grain_db, sample_id, block_id, grain_path, pixel_path, rel_Path):
    """
        解析文件，并存入晶粒数据库
        Parameters
        ------------
        graph py2neo数据库操作接口
        grain_db 晶粒图切割标签，用于区分不同切割下晶粒图谱
        sample_id 样本ID，用于晶粒的样本标签标识
        block_id 样品块ID，用于晶粒的样品的切块的标签标识
        grain_path,pixel_path,rel_Path atex文件的路径：晶粒数据文件，扫描点数据文件，相邻晶粒数据文件
    """
    if not (os.path.exists(grain_path) and os.path.exists(pixel_path) and os.path.exists(rel_Path)):
        raise Exception("\n警告！！ [%s]目录数据文件不全，解析终止!" % os.path.dirname(grain_path))
    # 1. 验证和初始化参数
    divide_num = eval_and_return_db_dividesize(grain_db) # 根据晶粒切割标签，验证晶粒划分规则，返回切割数量
    # 2.加载数据表
    df_of_grains = pd.read_csv(grain_path)          # 晶粒数据文件
    df_of_pixels = pd.read_csv(pixel_path)          # 晶粒扫描点文件
    df_of_rels = pd.read_csv(rel_Path, header=1)    # 相邻晶粒文件，首行是注释，不要
    # 解析数据
    max_axis = df_of_pixels.loc[:, ["X(p)", "Y(p)"]].max()  # 扫描点文件中，EBSD扫描点的最大坐标
    grain2len_dic = df_of_rels[["IDgrain", "BoundLengthTOT"]].set_index("IDgrain").to_dict()["BoundLengthTOT"] # 字典 {晶粒编号: 边长}，其他属性都在晶粒文件中，只有边长属性在相邻晶粒数据文件中，所以单独提炼出来
    # 3.先创建节点字典（防止插入数据库时，每次都生成一个新的节点。一个对象对应数据库中一个节点，如果创建多次，则会插入多个节点，插入关系时，不可以新建节点对象，应使用已有的。）
    nodes_dict = dict()
    for index_id, row in df_of_grains.iterrows():
        grain_id = row['#ID']
        if grain_id in nodes_dict.keys(): # 验证数据文件是否正确，存在重复晶粒ID重复
            raise Exception("数据文件中存在重复的晶粒！")
        # 构建晶粒节点（五个标签确定一个独立的晶粒节点：晶粒标签，ebsd切割方法标签，样本标签，样本切割块标签，ebsd切割方法下的切割子图标签）
        nodes_dict[grain_id] = Node("Grain", grain_db, "Sample"+sample_id, "Block"+block_id,
                        generate_sub_label(row[["Xg", "Yg"]], max_axis, divide_num),    # Sub label
                        grainID = grain_id,    boundLength = grain2len_dic[grain_id],   # attributes
                        phi1 = row["Phi1"],    phi = row["Phi"],     phi2 = row["Phi2"],# attributes
                        size = row["Size"],    surface = row["Surface"],                # attributes
                        angle = row["EllAng"], ellipticity = row["Ellip"],              # attributes
                        xAve = row["Xg"],      yAve = row["Yg"]                         # attributes
                      )
    # 4.创建关系，遍历晶粒相连表(注：index_id是dataframe的索引编号，不一定是当前列表中的数据顺序)
    for index_id, row in df_of_rels.iterrows():
        # 取出晶粒节点
        u_id, v_id = row[['IDgrain', 'IDneighbor']].astype(int)
        node_u, node_v = nodes_dict[u_id], nodes_dict[v_id]
        # 晶界属性
        grain_boundaries_of_u_and_v = {
            "boundLength": int(row["BoundLength"].astype(int)),
            "disorientation": float(row["Disorientation"])
        }
        # 创建关系（晶粒相连强度关系）
        bound_relation = generate_bound_relation(node_u, node_v,     # 首尾节点
                                                 grain_boundaries_of_u_and_v["boundLength"],    # 边属性
                                                 grain_boundaries_of_u_and_v["disorientation"]) # 边属性
        print("晶粒%d%s -[]-> 晶粒%d%s"
              % (node_u["grainID"], node_u.labels, node_v["grainID"], node_v.labels) )
        graph.create(bound_relation)


def file2neo4j(grain_db, filePath):
    """
        遍历晶粒数据文件，调用解析入库方法
        description: ！！！一个ebsd导出的三个文件必须放在一个文件夹下，该文件夹命名必须遵守样本[编号.切块编号]的规则，用于表示样本和切换标签！！！！其他上级文件夹命名可随意
        Param:  grain_db  晶粒切割方法标签
        Param:  filePath  所有文件根路径
    """
    for root, dirs, files in os.walk(filePath):
        for dir in dirs:
            if os.path.exists(os.path.join(root, dir, grain_filename)): # 如果该目录下包含我们要导入数据库的文件，取出目录名（小数点前为样本编号，小数点后为切块编号）。 目录固定格式
                sample_id, block_id = dir.split(".") # 样本编号,样本切块号(一个样本切为若干快，每块扫描一个EBSD)
                # 晶粒数据的三个文件
                grain_path = os.path.join(root, dir, grain_filename)
                pixel_path = os.path.join(root, dir, pixels_filename)
                rel_Path   = os.path.join(root, dir, relation_filename)
                parse_file_to_graph(global_graph, grain_db, sample_id, block_id, grain_path, pixel_path, rel_Path)
                # 多线程解析（不好用，neo4j社区版只支持单例，使用下面的代码将会导致程序突然结束）
                # _thread.start_new_thread(parse_file_to_graph,
                #                          (grain_dao.build_neo4j_connection(conf.db.uri, conf.db.username, conf.db.password),
                #                          grain_db, sample_id, block_id, grain_path, pixel_path, rel_Path))
def run():
    print("执行晶粒数据文件解析，开始导入数据库...，\n请确认导入前是否清空数据库？（重新导入输入1；不清空请回车，可能会重复节点）");
    op = input();
    if op == "1":
        # 清空数据库
        global_graph.delete_all()
    print("清空数据库成功！" if op=="1" else "不清空数据库。 ", "开始导入...")
    # 开始新建数据
    for grain_db in grain_dbs:
        file2neo4j(grain_db, file_path)

if __name__ == '__main__':
    run()