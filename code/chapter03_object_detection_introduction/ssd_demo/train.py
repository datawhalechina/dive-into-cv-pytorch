from layers.modules import MultiBoxLoss
from utils.my_dataset import VOC2012DataSet
from utils.config import Config
from ssd import build_ssd
import os
import sys
import time
import numpy as np
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data



def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print(device)
    VOC_root = parser_data.data_path
    batch_sizes = parser_data.batch_size

    Cuda = True

    model = build_ssd("train", Config["num_classes"])
    # 加载与训练模型
    print('Loading weights into state dict...')
    # device = torch.device(parser_data.device if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("/home/zdst/Ryan/pytorch-ssd-self/save_weights/ssd_weights.pth",
                                 map_location=device)  # 好像这个地方写device有error 后来直接换成cpu就可以了
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    annotation_path = '2012_train.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_train = len(lines)
    train_dataset = VOC2012DataSet(lines[:num_train], (Config["min_dim"],Config["min_dim"]))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True, drop_last=True, num_workers=4, collate_fn=collate_fn)

    #如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    # if parser_data.resume != "":
    #     print('Loading weights into state dict...')
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(parser_data.resume, map_location="cpu")
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    #     print('Finished!')
    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    #optimizer = torch.optim.Adam(params, lr=0.0005)
    #model.to(net)

    one_epoch_sizes = num_train // batch_sizes
    criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, Cuda)

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        if epoch%2 == 0:
            adjust_learning_rate(optimizer, 0.0005, 0.9, epoch)
        loc_loss = 0
        conf_loss = 0
        for iteration, dataset in enumerate(train_data_loader):
            if iteration >= one_epoch_sizes:
                break
            images, targets = dataset[0], dataset[1]
            with torch.no_grad():
                if Cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            #前向传播
            out = net(images)
            #梯度清零
            optimizer.zero_grad()
            #计算loss
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            #反向传播
            loss.backward()
            optimizer.step()

            #梯度加
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            print('\nEpoch:' + str(epoch+1) + '/' + str(parser_data.epochs))
            print('iter:' + str(iteration) + '/' + str(one_epoch_sizes) + '|| Loc_loss:%.4f || Conf_Loss:%.4f ||' % (loc_loss/(iteration+1), conf_loss/(iteration+1)), end=" ")

        print('Saving state, iter:', str(epoch+1))
        torch.save(model.state_dict(), 'save_weights/Epoch%d-Loc%.4f-Conf%.4f.pth'%((epoch+1), loc_loss/(iteration+1), conf_loss/(iteration+1)))

def adjust_learning_rate(optimizer, lr, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def collate_fn(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='/home/zdst/Ryan/dataset/voc_datasets/', help='dataset')
    # 文件保存地址
    parser.add_argument('--output-dir', default='/home/zdst/Ryan/pytorch-ssd-self/save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch_size大小
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
