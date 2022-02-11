
import math
import  torch
############# 节点特征 ###############

#### 晶粒结点--属性特征向量长度（自己定的晶粒各属性特征维度）
# `大小`
grain_node_size_feat_length=16
# `面积`
grain_node_surface_feat_length=16
# '长度'
grain_node_length_feat_length=16
# '欧拉角'
grain_node_per_eular_feat_length=16

# 个个特征拼接到一起，作为晶粒节点初始特征（目前使用的是0初始化）
grain_node_size = grain_node_size_feat_length+grain_node_surface_feat_length+grain_node_length_feat_length+grain_node_per_eular_feat_length
# grain_node_size = 200

#### 晶粒度结点--特征向量长度
# '晶粒度(目前是大小，取分割步长为2)'
grain_size_step = 3
grainsize_node_class_num = math.ceil(91.2/grain_size_step)  # 手动统计
grain_size_min = 0

#### 欧拉角结点--特征向量长度
# 每个欧拉角划分等级为`per_eular_class_num`,一共则有per_eular_class_num^3 个欧拉角结点
per_eular_class_num = 3  # 手动统计

grain_phi1_max = 223
grain_phi1_min = -82
# -24，-67，-35


grain_phi_max  = 180
grain_phi_min  = -1

grain_phi2_max = 107
grain_phi2_min = -61
#最小的还有 -61，-26, -24,-47,-37，-58，-47 但是不连续

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pth_path = 'pth_save'

