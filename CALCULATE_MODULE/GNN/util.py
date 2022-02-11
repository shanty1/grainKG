import hashlib
from pathlib import Path
from sklearn.metrics import r2_score
from CALCULATE_MODULE.GNN import config
import os
import time
import torch
import traceback

"""
    判断一个数在第几个区间内
"""
def get_num_space_idx(num,space_tuple_arrays):
    for i,(min,max) in enumerate(space_tuple_arrays):
        if num >= min and num < max:
            return i
        elif num == max and i == len(space_tuple_arrays)-1: # 容纳最后一个区间的终点
            return i
    raise Exception("\n{}不在[{:.2f},{:.2f}]范围内".format(num,space_tuple_arrays[0][0],space_tuple_arrays[-1][-1]))

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_util_dir():
    return os.path.dirname(__file__)

def save_model_dict2(model, model_name, epoch, loss, min_loss=200):
    date = time.strftime("-%Y-%m-%d")
    model_name = model_name + date + "(" + str(epoch) + "e)_dict.pth"
    save_path = os.path.join(get_util_dir(), config.pth_path)
    create_dir_not_exist(save_path)
    save_file = os.path.join(save_path, model_name)
    if loss < min_loss:
        torch.save(model.state_dict(), save_file)
        print("save model:%s, epoch:%s"%(model_name,epoch))
        min_loss = loss
    return min_loss

""" 保存模型和其他参数 """
def save_model_dict(model, optimizer, epoch, save_name, verbose=True):
    state = {'net': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    save_path = os.path.join(get_util_dir(), config.pth_path)
    create_dir_not_exist(save_path)
    save_file = os.path.join(save_path, save_name+str+".pth")
    torch.save(state, save_file)
    if verbose: print("save model:%s, epoch:%s"%(save_name,epoch))

""" 加载模型和其他参数 """
def load_model_from_dict(model_name, model, optimizer=None):
    path = os.path.join(get_util_dir(), config.pth_path, model_name + ".pth")
    if not os.path.exists(path):
        print("warning: no saved model! do nothing!")
        return 0
    else:
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['net'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = 0 if checkpoint['epoch']=="" else checkpoint['epoch']+ 1
        print("load model:%s" % (model_name.rsplit(".",1)[0]))
        return start_epoch

def load_model(model_name):
    save_path = os.path.join(get_util_dir(), config.pth_path)
    path = os.path.join(save_path, model_name)
    net = torch.load(path)
    print("load model:%s" % (model_name.rsplit(".",1)[0]))
    return net

def adjusted_r2(true, pred):
    r2 = r2_score(true, pred)
    adjusted_r2 = 1-((1-r2**2)*())

def md5(str):
    return hashlib.md5(str.encode(encoding='UTF-8')).hexdigest()

if __name__ == '__main__':
    i = md5("你好啊")
    print(type(i))