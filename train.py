import os
import re
import shutil
import time
from datetime import datetime, timedelta
import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from utils.getDevice import getDevice
from utils.getData import CustomDataset
from utils.resnet import resnet34
from utils.getConfig import getConfig

config = getConfig("config.yaml")
gpus = str(config['train']['gpus'])
batch_size = int(config['train']['batch_size'])
num_workers = int(config['train']['num_workers'])
num_epoch = int(config['train']['num_epoch'])
learning_rate = float(config['train']['learning_rate'])
train_list_path = str(config['train']['train_list_path'])
test_list_path = str(config['train']['test_list_path'])
save_model = str(config['train']['save_model'])
resume = bool(config['train']['resume'])
pretrained_model = str(config['train']['pretrained_model'])


@torch.no_grad()
def modelTest(model, test_loader, device):
    accuracies = []
    for spec_mel, label in test_loader:
        spec_mel = spec_mel.to(device)
        label = label.to(device)
        pred = model(spec_mel)
        pred = pred.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = label.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))
        accuracies.append(acc.item())
    return float(sum(accuracies) / len(accuracies)) if len(accuracies) != 0 else float(0)


# 保存模型
def saveModel(model, optimizer, epoch_id):
    model_params_path = os.path.join(save_model, 'epoch_%d' % epoch_id)
    if not os.path.exists(model_params_path):
        os.makedirs(model_params_path)
    # 保存模型参数和优化方法参数
    torch.save(model.state_dict(), os.path.join(model_params_path, 'model_params.pth'))
    torch.save(optimizer.state_dict(), os.path.join(model_params_path, 'optimizer.pth'))
    # 删除旧的模型
    old_model_path = os.path.join(save_model, 'epoch_%d' % (epoch_id - 3))
    if os.path.exists(old_model_path):
        shutil.rmtree(old_model_path)
    # 保存整个模型和参数
    all_model_path = os.path.join(save_model, 'resnet34.pth')
    if not os.path.exists(os.path.dirname(all_model_path)):
        os.makedirs(os.path.dirname(all_model_path))
    torch.jit.save(torch.jit.script(model), all_model_path)


def resumeTrain(optimizer, device_ids, model):
    pretrained_model_path = os.path.join(save_model, pretrained_model)
    optimizer_state = torch.load(os.path.join(pretrained_model_path, 'optimizer.pth'))
    optimizer.load_state_dict(optimizer_state)
    # 获取预训练的epoch数
    last_epoch = int(re.findall('(\d+)', pretrained_model_path)[-1])
    if len(device_ids) > 1:
        model.module.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'model_params.pth')))
    else:
        model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'model_params.pth')))
    print('成功加载模型参数和优化方法参数')
    return optimizer, model, last_epoch


def train():
    device_ids = [int(i) for i in gpus.split(',')]
    device = getDevice()
    # 获取数据集
    train_dataset = CustomDataset(train_list_path)
    # 加载我们的数据集
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size * len(device_ids),
                              shuffle=True,
                              num_workers=num_workers)
    # 这边是加载我们的测试数据集
    test_dataset = CustomDataset(test_list_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers)
    # 构建restNet模型（残差结构网络），这个网络可以可以图像分类
    model = resnet34()
    # 如果有多个GPU，那么就多个GPU一起训练
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    # 首先加载我们的模型，然后打印一下模型的结构
    model.to(device)
    summary(model, (1, 257, 257))
    # 初始化epoch数
    last_epoch = 0
    # 获取优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 获取损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 加载模型参数和优化方法参数
    # 这里是加载一下预训练模型
    if resume:
        optimizer, model, last_epoch = resumeTrain(optimizer, device_ids, model)
    # 开始训练
    sum_batch = len(train_loader) * (num_epoch - last_epoch)
    for epoch_id in range(last_epoch, num_epoch):
        epoch_id += 1
        batch_id = 0
        # 获取我们的输入和标签
        for spec_mel, label in train_loader:
            batch_id += 1
            start = time.time()
            spec_mel = spec_mel.to(device)
            label = label.to(device)
            # 先调用restNet获取特征值
            pred = model(spec_mel)
            # 然后根据标签和特征值计算输出
            # output = metric_fc(feature, label)
            # 计算loss
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每迭代10次就打印一下准确率和loss信息
            if batch_id % 10 == 0:
                pred = pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((pred == label).astype(int)).item()
                eta_sec = ((time.time() - start) * 1000) * (
                        sum_batch - (epoch_id - last_epoch) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f, lr: %f, eta: %s' % (
                    datetime.now(), epoch_id, batch_id, len(train_loader), loss.item(), acc, learning_rate,
                    eta_str))
        # 迭代完一轮后就对我们的模型进行评估
        # 开始评估
        model.eval()
        print('=' * 70)
        accuracy = modelTest(model, test_loader, device)
        model.train()
        print('[{}] Test epoch {} Accuracy {:.5}'.format(datetime.now(), epoch_id, accuracy))
        print('=' * 70)

        # 保存模型
        if len(device_ids) > 1:
            saveModel(model.module, optimizer, epoch_id)
        else:
            saveModel(model, optimizer, epoch_id)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    train()