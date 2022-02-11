import os
import time
import dgl
import torch as torch
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import copy

from torch import tensor

import config
from CALCULATE_MODULE.GNN.my_dataset import get_dataset, EbsdDataset
import util
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from CALCULATE_MODULE.visulization.figure_helper import plot_fiting_points

device = config.device


class Trainer():
    """
        模型训练器
        :Parameters
        --------------
        model_name  模型名称（同时是模型保存的文件名称，会覆盖已存在文件）
             model  网络模型
          trainset  训练集，EbsdDataset 类型 （注：此训练集包含训练和验证数据）
          test_set  测试集，EbsdDataset 类型
         optimizer  优化器
         scheduler  学习率调整器，Default: None
         random_state   还原训练过程
         savepath   模型存储路径（相对于util根目录，即GNN目录下）
    """
    def __init__(self, model_name, model, trainset, test_set, optimizer, scheduler=None, random_state=None, save_path=config.pth_path):
        self.name = model_name
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset_for_train_val = trainset
        self.random_state = random_state
        self.dataset_for_test = test_set
        self.loss_func = torch.nn.MSELoss()
        self.k_models = None    # 最优k个模型
        self.save_file = os.path.join(util.get_util_dir(), save_path, model_name + ".pth")
        util.create_dir_not_exist(os.path.join(util.get_util_dir(), save_path))

    def train(self, num_epochs=1000, k_fold=5, batch_size=999, kout=False):
        """
            训练模型
            :param  k_fold: 交叉验证折数 value -> [1, dataset.size]，
                            留一法：value = 1 or dataset.size
                            默认：5
            :param  num_epochs: 迭代次数
            :param  kout:  打印k折交叉验证过程
        """
        if k_fold == 1: k_fold = len(self.dataset_for_train_val)
        kFold = KFold(n_splits=k_fold, shuffle=True, random_state=self.random_state)

        since = time.time()
        lowest_loss = float('inf')

        for epoch in range(num_epochs):
            print('Epoch {}/{} (lr={})'.format(epoch, num_epochs - 1, self.optimizer.state_dict()['param_groups'][0]['lr']))
            print('-' * 10)

            # 当前epoch的loss记录（每折交叉验证下的loss累加）
            epoch_loss = {"train": 0.0, "val": 0.0}
            k_models = []

            # ------------------------------------------------- K折交叉训练验证 ----------------------------------------------------
            for i, (train_index, test_index) in enumerate (kFold.split(self.dataset_for_train_val.labels)):
                (x_train, y_train), (x_valid, y_valid) \
                    = (([self.dataset_for_train_val.graphs[i] for i in index],
                        self.dataset_for_train_val.labels[index])
                       for index in [train_index, test_index])
                k_datasets = {
                    "train": EbsdDataset(x_train, y_train, force_reload=True),
                    "val"  : EbsdDataset(x_valid, y_valid, force_reload=True)
                }
                k_dataloaders = {type: GraphDataLoader(k_datasets[type], batch_size=batch_size, shuffle=True) for type in ["train", "val"]}

                # ------------------------------------------------- 轮训训练、验证 ----------------------------------------------------
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()  # Set model to evaluate mode

                    #------------------------------------------------- 预测dataloaders的结果 ----------------------------------------------------
                    running_loss = 0.0  # 叠加dataloader的loss
                    for inputs, labels in k_dataloaders[phase]:
                        inputs, labels = inputs.to(device), labels.to(device)

                        self.optimizer.zero_grad()                   # zero the parameter gradients
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)             # 预测
                            loss = self.loss_func(outputs, labels)   # loss
                            if phase == 'train':                # backward + optimize
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * labels.size(0)

                    # 打印第 i/k折 phase=['train','val']的 loss
                    if kout:
                        i_fold_loss = running_loss / len(k_datasets[phase])     # Fold[i/k] loss
                        if phase == "train":
                            if i!=0: print()
                            print('KFold {}/{}\t'.format(i+1, k_fold), end="  ")
                        print('{}：{:.3f}  '.format(phase, i_fold_loss), end="")

                    # 累加epoch的总loss epoch_loss += loss[i/k折]
                    epoch_loss[phase] += running_loss

                # 训练和验证都结束 (存入当前k折训练的models)
                k_models.append(copy.deepcopy(self.model))

            # epoch循环下：更新学习率 (一个epoch下的k折全部结束)
            if self.scheduler: self.scheduler.step()

            # 打印 epoch loss
            epoch_loss["train"] = epoch_loss["train"] / (len(self.dataset_for_train_val) * (k_fold - 1))
            epoch_loss["val"]   = epoch_loss["val"]   / len(self.dataset_for_train_val)
            print("\n\nTrain loss:%.3f" % epoch_loss["train"])
            print("  val loss:%.3f" % epoch_loss["val"])
            print()

            # 保存优化模型
            if epoch_loss["val"] < lowest_loss:
                lowest_loss = epoch_loss["val"]                             # 更新最小loss
                self.k_models = k_models
                self.save_k_models(k_models)

        end = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(end // 60, end % 60))
        print('Best val loss: {:.3f}'.format(lowest_loss))

    def test(self):
        """
            reverse_scaler 显示真实值所需的训练集归一化器，如果为None，则打印归一化值（需要是训练集fit后的）
        """
        if self.k_models or self.load_k_models(): # 优先使用内置k_models
            inputs = dgl.batch(self.dataset_for_test.graphs)
            labels = self.dataset_for_test.labels
            outputs = self.__k_predict__(inputs)
            # index
            mse = mean_squared_error(labels, outputs)
            mae = mean_absolute_error(labels, outputs)
            r2 = r2_score(labels, outputs)
            ev = explained_variance_score(labels, outputs)
            print("mse\t\t mae\t\t ev\t\t r2\t\t\n"
                  "{:.3f}\t {:.3f}\t\t {:.3f}\t {:.3f}"
                  .format(mse, mae, ev, r2))

            if self.dataset_for_train_val.fited_scaler_y:    # 真实标签和预测值
                outputs = self.dataset_for_train_val.fited_scaler_y.inverse_transform(outputs)
                labels = self.dataset_for_train_val.fited_scaler_y.inverse_transform(labels)
            print("预测值：", *("%10.2f" % o.item() for o in outputs.flatten()))
            print("真实值：", *("%10.2f" % l.item() for l in labels.flatten()) ,end="\n\n")
            plot_fiting_points(outputs.flatten(), labels.flatten(),
                               title="R2 = %.3f" % r2,
                               savefile="{}/figures/{}.png".format(os.path.dirname(__file__),self.name))
        else:
            raise Exception("train trainer first !")

    def __k_predict__(self, inputs, return_feat=False):
        with torch.set_grad_enabled(False):
            for model in self.k_models:
                model.eval()
                model = model.cpu()
            k_outputs = torch.stack([model(inputs,return_feat) for model in self.k_models])
            return k_outputs.mean(axis=0)

    def predict(self, inputs, return_feat=False):
        if self.k_models or self.load_k_models(): # 优先使用内置k_models
            if isinstance(inputs, list):
                inputs = dgl.batch(inputs)
            outputs = self.__k_predict__(inputs, return_feat)
            if not return_feat and self.dataset_for_train_val.fited_scaler_y:  # 真实标签和预测值
                outputs = self.dataset_for_train_val.fited_scaler_y.inverse_transform(outputs)
            return outputs
        else:
            raise Exception("train trainer first !")

    def save_k_models(self, k_models):
        state = {'k_models': [m.state_dict() for m in k_models],
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.save_file)

    def load_k_models(self):
        if not os.path.exists(self.save_file):
            print("warning: want to load k_models. but no saved model! do nothing!")
            return False
        else:
            checkpoint = torch.load(self.save_file, map_location='cpu')
            self.k_models = []
            for dict in checkpoint['k_models']:
                self.k_models.append(copy.deepcopy(self.model))
                self.k_models[-1].load_state_dict(dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("load model:%s" % self.name)
            return True
