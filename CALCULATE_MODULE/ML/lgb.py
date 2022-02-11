import lightgbm as lgb
from lightgbm import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pylab import mpl
import warnings
# filter warnings
warnings.filterwarnings('ignore')
# 正常显示中文

mpl.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示符号
from matplotlib import rcParams

rcParams['axes.unicode_minus'] = False
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

scale_x = StandardScaler()
scale_y = StandardScaler()

def loadXY(datafilepath, label_flag = '标签名'):
    # data = pd.read_table(datafilepath, sep=',')
    data = pd.read_csv(datafilepath)
    x = data.loc[:, data.columns != label_flag]
    y = data.loc[:, label_flag]

    mean_cols = x.mean()
    # x=x.fillna(mean_cols)  #填充缺失值
    # x=pd.get_dummies(x)    #独热编码
    # y = np.log(y)  # 平滑处理Y
    y = np.array(y).reshape(-1, 1)
    # 归一化
    # mm_x = MinMaxScaler()
    # x = mm_x.fit_transform(x)
    # 标准化
    x = scale_x.fit_transform(x)
    # y = scale_y.fit_transform(y)

    y = y.ravel()  # 转一维
    return x, y



# seeds=[i for i in range(100)]
seed = None
datafilepath = './data/cleanData.'
label_flag = 'categorical'
test_size = 0.4
if __name__ == '__main__':
    x, y = loadXY(datafilepath, label_flag)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=seed, shuffle=True)

    # '''
    train_data = Dataset(x_train, label=y_train)
    val_data = Dataset(x_val, label=y_val)
    test_data  = Dataset(x_test,  label=y_test)

    param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary', 'metric': ['auc', 'binary_logloss']}
    num_round = 10
    bstcv = lgb.cv(param, train_data, num_round, nfold=10)
    bst = lgb.train(param, train_data, num_round, valid_sets=[val_data], early_stopping_rounds=10)
    bst.save_model('model.txt', num_iteration=bst.best_iteration)
    test_pred = bst.predict(x_test, num_iteration=bst.best_iteration)
    test_pred = np.rint(test_pred)
    accuracy = accuracy_score(y_test, test_pred)  # 准确度
    precision = precision_score(y_test, test_pred)  # 精确度
    recall = recall_score(y_test, test_pred)  # 召回率
    macro_f1 = f1_score(y_test, test_pred, average='macro')
    micro_f1 = f1_score(y_test, test_pred, average='micro')
    weighted_f1 = f1_score(y_test, test_pred, average='weighted')
    print('{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}'.format('正例', '准确率', '精确率', '召回率', 'macro-f1', 'micro-f1',
                                                              'weighted-f1'))
    print('{:10s}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}'.format('就业', accuracy, precision, recall,
                                                                          macro_f1, micro_f1, weighted_f1))

    accuracy = accuracy_score(y_test, test_pred)  # 准确度
    precision = precision_score(y_test, test_pred, pos_label=0)  # 精确度
    recall = recall_score(y_test, test_pred, pos_label=0)  # 召回率
    macro_f1 = f1_score(y_test, test_pred, average='macro', pos_label=0)
    micro_f1 = f1_score(y_test, test_pred, average='micro', pos_label=0)
    weighted_f1 = f1_score(y_test, test_pred, average='weighted', pos_label=0)
    print('{:10s}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}{:10.4f}'.format('未就业', accuracy, precision, recall,
                                                                          macro_f1, micro_f1, weighted_f1))
    # '''