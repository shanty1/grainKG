# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import r2_score, mean_absolute_error
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torchvision.models import Inception3

from CALCULATE_MODULE.CNN.EbsdImageSet import get_ebsd_image_sets, EBSDImageSet


def seed_torch(seed=1423):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# seed_torch()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plt.ion()  # interactive mode

mean = np.array([0, 0, 0])
std = np.array([1, 1, 1])

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomCrop((320, 270)),  # 665*501
        transforms.CenterCrop((320, 270)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        # transforms.Resize((325, 244)),
        transforms.CenterCrop((320,270)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
}

data_dir = 'image_data/subgraph4'
images, labels = get_ebsd_image_sets(1)

image_datasets = {x: EBSDImageSet(images[x],labels[x],data_transforms[x]) for x in ['train', 'test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=20,
                                              shuffle=True, num_workers=6)
               for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

label_size = len(labels["train"][0])

# device = torch.device("cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, save_name='parameter'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['test', 'train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                labels = labels.to(device)
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds, pos = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                if scheduler: scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'test' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # save best model weights
                torch.save(model.state_dict(), './save/' + save_name + '.pkl')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test loss: {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds, pos = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('pred : {}, label:{}'.format(preds[j], labels[j]), loc="left")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_train_data():
    # Get a batch of training data
    inputs, labels = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs, padding=9, pad_value=9.0)

    imshow(out, title=labels)


def model(net_name):
    if 'resnet34' in net_name:
        model_ft = models.resnet34(pretrained=True)
    elif 'resnet50' in net_name:
        model_ft = models.resnet50(pretrained=True)
    elif 'vgg19' in net_name:
        model_ft = models.vgg19(pretrained=True)
    elif 'densenet201' in net_name:
        model_ft = models.densenet201(pretrained=True)
    elif 'densenet161' in net_name:
        model_ft = models.densenet161(pretrained=True)
    elif 'inception_v3' in net_name:
        model_ft = models.inception_v3(pretrained=True)

    for param in model_ft.parameters():
        param.requires_grad = False
        pass

    if 'resnet' in net_name:
        num_ftrs = model_ft.fc.in_features
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, label_size),
        )
    elif 'densenet' in net_name:
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs, label_size),
        )
    elif 'vgg' in net_name:
        model_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, label_size),
        )
    return model_ft


def train(net_name):
    show_train_data()
    model_ft = model(net_name).to(device)
    # load model
    if os.path.exists('./save/' + net_name + '.pkl'):
        dict = torch.load('./save/' + net_name + '.pkl')
        model_ft.load_state_dict(dict)
    criterion = nn.MSELoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.6)
    exp_lr_scheduler = None

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=300, save_name=net_name)
    # visualize_model
    # visualize_model(model_ft)


def predict(net_name='parameter'):
    # load data
    inputs, labels = next(iter(dataloaders['test']))
    # load model
    dict = torch.load('./save/' + net_name + '.pkl', map_location='cpu')
    model_ft = model(net_name)
    model_ft.load_state_dict(dict)
    # track history if only in train
    model_ft.eval()  # Set model to evaluate mode
    criterion = nn.MSELoss()
    with torch.set_grad_enabled(False):
        preds = model_ft(inputs)
        r2 = r2_score(labels,preds)
        mse = criterion(preds, labels)
        mae = mean_absolute_error(labels,preds)
    # draw fitting performance result
    # plt.text(120, 80, 'loss:'+ str(loss.item()))
    # print(preds.numpy().tolist())
    # print(labels.numpy().tolist())
    print(net_name,"\n r2:{:.4f} mse:{:.4f} mae:{:.4f} ".format(r2, mse, mae))
    plt.scatter(labels, preds, label=net_name+', loss: {:.0f}'.format(mse.item()))
    c = torch.cat((labels, preds), 0)
    plt.plot(c, c)
    plt.xlabel('label')
    plt.ylabel('prediction')


if __name__ == '__main__':

    # @ predict
    # predict('ResNet50')

    # # @ train
    net_names = ['resnet34', 'resnet50','densenet161','densenet201' ]
    for net_name in net_names:
        train(net_name)
        predict(net_name)

    plt.legend()

    plt.ioff()
    plt.show()
