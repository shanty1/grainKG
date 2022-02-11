import _thread
import os
import random
import numpy as np
import torch
import torchvision
from bitarray._util import subset
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from CALCULATE_MODULE.CNN.EbsdImageSet import get_ebsd_image_sets, EBSDImageSet
from CALCULATE_MODULE.CNN.cnn_trainer import Trainer

SEED = 1423

def setup_seed(seed):
    global SEED
    SEED = seed
    if seed == None:
        return
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(SEED)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    torch.backends.cudnn.deterministic = True


def pre_model(net_name, out_dim, train_all_parameters=True, pretrained=False):
    if 'resnet34' in net_name:
        model_ft = models.resnet34(pretrained=pretrained)
    elif 'resnet50' in net_name:
        model_ft = models.resnet50(pretrained=pretrained)
    elif 'vgg11' in net_name:
        model_ft = models.vgg11(pretrained=pretrained)
    elif 'vgg16' in net_name:
        model_ft = models.vgg16(pretrained=pretrained)
    elif 'vgg19' in net_name:
        model_ft = models.vgg19(pretrained=pretrained)
    elif 'densenet201' in net_name:
        model_ft = models.densenet201(pretrained=pretrained)
    elif 'densenet161' in net_name:
        model_ft = models.densenet161(pretrained=pretrained)
    elif 'densenet121' in net_name:
        model_ft = models.densenet121(pretrained=pretrained)
    elif 'inception_v3' in net_name:
        model_ft = models.inception_v3(pretrained=pretrained)

    if not train_all_parameters:
        for param in model_ft.parameters():
            param.requires_grad = False

    if net_name.startswith("resnet") or net_name.startswith("inception"):
        num_ftrs = model_ft.fc.in_features
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_dim),
        )
    elif 'densenet' in net_name:
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_dim),
        )
    elif 'vgg' in net_name:
        model_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_dim),
        )
    return model_ft


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(str(title))
    plt.show()

def show_train_data(images_tensor, labels):
    # Make a grid from batch
    out = torchvision.utils.make_grid(images_tensor,nrow=4, padding=9, pad_value=9.0)
    imshow(out, title=labels)

def cal_mean_std(image_datasets):
    means = []
    stds = []
    for img in image_datasets['train']:
        means.append(torch.mean(img))
        stds.append(torch.std(img))

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))

mean=[0.4845, 0.4541, 0.4025]	,
std=[0.2724, 0.2637, 0.2761]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
}

def build_trainer(name):
    images, labels = get_ebsd_image_sets(test_size=0.2, target="block",
                                         label_options=["ys", "uts", "el"], directions=["90"],
                                         dir_path="./image_data/subgraph4", random_state=SEED)
    image_datasets = {x: EBSDImageSet(images[x], labels[x], data_transforms[x]) for x in ['train', 'test']}

    model = pre_model(name, out_dim=len(labels["train"][0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD (model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = None #
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.65)

    trainer = Trainer(model_name = name,
                       model = model,
                       trainset = image_datasets["train"],
                       test_set = image_datasets["test"],
                       optimizer = optimizer,
                       scheduler = scheduler,
                       random_state=SEED)

    show_train_data(*image_datasets["train"][:12])

    return trainer


"""
    【作者提示】：
        1. 对比训练请固定随机种子$SEED$，并不要更换。随机种子将影响数据集的划分、模型参数训练。
        2. 由于数据量小，使用留一法训练能达到最大的拟合性能。
"""

if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(1423)
    trainers = (          # name
        build_trainer("densenet121"),  #
        build_trainer("inception_v3"),  #
        build_trainer("resnet50"), #
        build_trainer("resnet34"), #
    )

    for trainer in trainers:                     # epoch  k batch
        # trainer.train(num_epochs=50, k_fold=10, batch_size=20)
        pass

    for trainer in trainers:
        trainer.test()
