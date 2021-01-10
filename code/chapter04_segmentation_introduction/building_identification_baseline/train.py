# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from optparse import OptionParser

import torch
import torch.nn as nn
from torch import optim

from unet import UNet
from dataset import load_dataset_info, split_dataset_info, Building_Dataset
from utils.model_saver import ModelSaver


def train_net(net, paras):
    # parameters
    img_dir = paras.image_dir
    anno_path = paras.anno_path
    checkpoint_dir = paras.model_save_dir
    val_percent = 0.1
    epochs = paras.epochs
    batch_size = paras.batch_size
    lr = paras.learning_rate
    num_workers = 2

    # torch model saver
    saver = ModelSaver(max_save_num = 5)

    # load dataset info
    dataset = load_dataset_info(img_dir, anno_path)
    train_set_info, valid_set_info = split_dataset_info(dataset, val_percent)

    # build dataloader
    building_trainset = Building_Dataset(train_set_info)
    building_validset = Building_Dataset(valid_set_info)
    train_dataloader = torch.utils.data.DataLoader(building_trainset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    valid_dataloader = torch.utils.data.DataLoader(building_validset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    # optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # loss function
    #criterion = nn.L1Loss(reduce=True, size_average=True)
    criterion = nn.BCELoss() 


    train_num = len(building_trainset)
    valid_num = len(building_validset)
    print('''
    Starting training:
        Total Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints save dir: {}
    '''.format(epochs, batch_size, lr, train_num, valid_num, checkpoint_dir))

    # ------------------------
    # start training...
    # ------------------------
    best_valid_loss = 1000
    for epoch in range(1, epochs+1):
        print('Starting epoch {}/{}.'.format(epoch, epochs))

        # training
        net.train()
        epoch_loss = 0
        for idx, data in enumerate(train_dataloader):
            imgs, true_masks = data

            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

            pred_masks = net(imgs)

            # compute loss
            loss = criterion(pred_masks, true_masks)
            epoch_loss += loss.item()

            if idx % 10 == 0:
                print(f'{idx}/{len(train_dataloader)}, loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / len(train_dataloader)
        print('Epoch finished ! Loss: {}\n'.format(epoch_loss))


        # validation
        net.eval()
        valid_loss = 0
        with torch.no_grad():
            for idx, data in enumerate(valid_dataloader):
                if idx % 10 == 0:
                    print(idx, '/', len(valid_dataloader))
                imgs, true_masks = data

                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

                # inference
                pred_masks = net(imgs)
                # compute loss
                loss = criterion(pred_masks, true_masks)
                valid_loss += loss.item()
        valid_loss = valid_loss / len(valid_dataloader)
        print('Validation finished ! Loss:{}  Best Loss before:{}\n'.format(valid_loss, best_valid_loss))


        # save check_point
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('New best model find, Checkpoint {} saving...'.format(epoch))
            model_save_path = os.path.join(checkpoint_dir, '{}_CP{}.pth'.format(best_valid_loss, epoch))
            #torch.save(net.state_dict(), model_save_path)
            saver.save_new_model(net, model_save_path)


def get_args():
    parser = OptionParser()
    parser.add_option('-d', '--gpu_devices', dest='gpu_devices',
                      default='0', help='use which cuda device')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load pretrain model to continue training or not')
    parser.add_option('-p', '--pretrain_path', dest='pretrain_path',
                      default="checkpoint/*.pth", help='the path of pretrained model.')
    parser.add_option('-s', '--model_save_dir', dest='model_save_dir',
                      default='checkpoint', help='directory to save checkpoint')
    parser.add_option('-i', '--image', dest='image_dir',
                      default='./data/train', help='directory of train image')
    parser.add_option('-m', '--anno', dest='anno_path',
                      default='./data/train_mask.csv', help='train mask anno path')
    parser.add_option('-e', '--epochs', dest='epochs', default=30, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--bs', dest='batch_size', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--lr', dest='learning_rate', default=0.001,
                      type='float', help='learning rate')

    (options, args) = parser.parse_args()
    return options



if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    device = torch.device('cuda')

    net = UNet(n_channels=3, n_map=1)
    net.to(device)

    if args.load == 'True':
        net.load_state_dict(torch.load(args.pretrain_path))
        print('loade model from {}'.format(args.pretrain_path))

    train_net(net=net, paras=args)
