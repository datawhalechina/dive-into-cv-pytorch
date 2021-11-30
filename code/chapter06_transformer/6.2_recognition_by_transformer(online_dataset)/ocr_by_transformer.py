# -*- coding: utf-8 -*-
"""

use transformer to do OCR!

利用transformer来完成一个简单的OCR字符识别任务

@author: anshengmath@163.com
"""
import os
import time
import copy
from PIL import Image

from tensorbay import GAS
from tensorbay.dataset import Dataset

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from analysis_recognition_dataset import load_lbl2id_map, statistics_max_len_label
from transformer import *
from train_utils import *


class Recognition_Dataset(object):

    def __init__(self, segment, lbl2id_map, sequence_len, max_ratio, pad=0):

        self.data = segment
        self.lbl2id_map = lbl2id_map
        self.pad = pad   # padding标识符的id，默认0
        self.sequence_len = sequence_len    # 序列长度
        self.max_ratio = max_ratio * 3      # 将宽拉长3倍

        # 定义随机颜色变换
        self.color_trans = transforms.ColorJitter(0.1, 0.1, 0.1)
        # 定义 Normalize
        self.trans_Normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])

    def __getitem__(self, index):
        """
        获取对应index的图像和ground truth label，并视情况进行数据增强
        """
        img_data = self.data[index]
        lbl_str = img_data.label.classification.category  # 标签

        # ----------------
        # 图片预处理
        # ----------------
        # load image
        with img_data.open() as fp:
            img = Image.open(fp).convert('RGB')

        # 对图片进行大致等比例的缩放
        # 将高缩放到32，宽大致等比例缩放，但要被32整除
        w, h = img.size
        ratio = round((w / h) * 3)   # 将宽拉长3倍，然后四舍五入
        if ratio == 0:
            ratio = 1
        if ratio > self.max_ratio:
            ratio = self.max_ratio
        h_new = 32
        w_new = h_new * ratio
        img_resize = img.resize((w_new, h_new), Image.BILINEAR)

        # 对图片右半边进行padding，使得宽/高比例固定=self.max_ratio
        img_padd = Image.new('RGB', (32*self.max_ratio, 32), (0,0,0))
        img_padd.paste(img_resize, (0, 0))

        # 随机颜色变换
        img_input = self.color_trans(img_padd)
        # Normalize
        img_input = self.trans_Normalize(img_input)

        # ----------------
        # label处理
        # ----------------

        # 构造encoder的mask
        encode_mask = [1] * ratio + [0] * (self.max_ratio - ratio)
        encode_mask = torch.tensor(encode_mask)
        encode_mask = (encode_mask != 0).unsqueeze(0)

        # 构造ground truth label
        gt = []
        gt.append(1)    # 先添加句子起始符
        for lbl in lbl_str:
            gt.append(self.lbl2id_map[lbl])
        gt.append(2)
        for i in range(len(lbl_str), self.sequence_len):   # 除去起始符终止符，lbl长度为sequence_len，剩下的padding
            gt.append(0)
        # 截断为预设的最大序列长度
        gt = gt[:self.sequence_len]

        # decoder的输入
        decode_in = gt[:-1]
        decode_in = torch.tensor(decode_in)
        # decoder的输出
        decode_out = gt[1:]
        decode_out = torch.tensor(decode_out)
        # decoder的mask
        decode_mask = self.make_std_mask(decode_in, self.pad)
        # 有效tokens数
        ntokens = (decode_out != self.pad).data.sum()

        return img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens


    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        padd 和 future words 均在mask中用0表示
        """
        tgt_mask = (tgt != pad)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = tgt_mask.squeeze(0)   # subsequent返回值的shape是(1, N, N)
        return tgt_mask

    def __len__(self):
        return len(self.data)


# Model Architecture
class OCR_EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, src_position, tgt_embed, generator):
        super(OCR_EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed    # input embedding module(input embedding + positional encode)
        self.src_position = src_position
        self.tgt_embed = tgt_embed    # ouput embedding module
        self.generator = generator    # output generation module

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        res = self.decode(memory, src_mask, tgt, tgt_mask)
        return res

    def encode(self, src, src_mask):
        # feature extract
        src_embedds = self.src_embed(src)
        # 将src_embedds由shape(bs, model_dim, 1, max_ratio) 处理为transformer期望的输入shape(bs, 时间步, model_dim)
        src_embedds = src_embedds.squeeze(-2)
        src_embedds = src_embedds.permute(0, 2, 1)

        # position encode
        src_embedds = self.src_position(src_embedds)

        return self.encoder(src_embedds, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        target_embedds = self.tgt_embed(tgt)
        return self.decoder(target_embedds, memory, src_mask, tgt_mask)


def make_ocr_model(tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建模型
    params:
        tgt_vocab: 输出的词典大小(82)
        N: 编码器和解码器堆叠基础模块的个数
        d_model: 模型中embedding的size，默认512
        d_ff: FeedForward Layer层中embedding的size，默认2048
        h: MultiHeadAttention中多头的个数，必须被d_model整除
        dropout:
    """
    c = copy.deepcopy

    backbone = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])    # 去掉最后两个层 (global average pooling and fc layer)

    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = OCR_EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        backbone,
        c(position),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # Initialize parameters with Glorot / fan_avg.
    for child in model.children():
        if child is backbone:
            # 将backbone的权重设为不计算梯度
            for param in child.parameters():
                param.requires_grad = False
            # 预训练好的backbone不进行随机初始化，其余模块进行随机初始化
            continue
        for p in child.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model


def run_epoch(data_loader, model, loss_compute, device=None):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_loader):
        #if device == "cuda":
        #    batch.to_device(device)
        img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)
        decode_in = decode_in.to(device)
        decode_out = decode_out.to(device)
        decode_mask = decode_mask.to(device)
        ntokens = torch.sum(ntokens).to(device)

        out = model.forward(img_input, decode_in, encode_mask, decode_mask)

        loss = loss_compute(out, decode_out, ntokens)
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# greedy decode
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    memory = model.encode(src, src_mask)
    # ys代表目前已生成的序列，最初为仅包含一个起始符的序列，不断将预测结果追加到序列最后
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long()
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        next_word = torch.ones(1, 1).type_as(src.data).fill_(next_word).long()
        ys = torch.cat([ys, next_word], dim=1)

        next_word = int(next_word)
        if next_word == end_symbol:
            break
        #ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    ys = ys[0, 1:]
    return ys


def judge_is_correct(pred, label):
    # 判断模型预测结果和label是否一致
    pred_len = pred.shape[0]
    label = label[:pred_len]
    is_correct = 1 if label.equal(pred) else 0
    return is_correct


if __name__ == "__main__":

    # TODO set parameters
    device = torch.device('cuda')  # cpu 或 cuda
    nrof_epochs = 1500
    batch_size = 64
    model_save_path = './log/ex1_ocr_model.pth'

    # GAS凭证
    KEY = 'Accesskey-fd26cc098604c68a99d3bf7f87cd480a'  # 添加自己的AccessKey
    gas = GAS(KEY)
    # 在线获取数据集
    dataset_online = Dataset("ICDAR2015", gas)
    dataset_online.enable_cache('./data')  # 数据本地缓存

    # 在线获取训练集和验证集
    train_data_online = dataset_online["train"]
    valid_data_online = dataset_online['valid']

    # 读取label-id映射关系记录文件
    lbl2id_map_path = os.path.join('./', 'lbl2id_map.txt')
    lbl2id_map, id2lbl_map = load_lbl2id_map(lbl2id_map_path)

    # 统计数据集中出现的所有的label中包含字符最多的有多少字符，数据集构造gt信息需要用到
    train_max_label_len = statistics_max_len_label(train_data_online)
    valid_max_label_len = statistics_max_len_label(valid_data_online)
    sequence_len = max(train_max_label_len, valid_max_label_len)   # 数据集中字符数最多的一个case作为制作的gt的sequence_len

    # 构造 dataloader
    max_ratio = 8    # 图片预处理时 宽/高的最大值，不超过就保比例resize，超过会强行压缩
    train_dataset = Recognition_Dataset(train_data_online, lbl2id_map, sequence_len, max_ratio, pad=0)
    valid_dataset = Recognition_Dataset(valid_data_online, lbl2id_map, sequence_len, max_ratio, pad=0)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)  # cpu -> 0
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4)  # cpu -> 0

    # build model
    # use transformer as ocr recognize model
    tgt_vocab = len(lbl2id_map.keys())
    d_model = 512
    ocr_model = make_ocr_model(tgt_vocab, N=5, d_model=d_model, d_ff=2048, h=8, dropout=0.1)
    ocr_model.to(device)

    # train prepare
    criterion = LabelSmoothing(size=tgt_vocab, padding_idx=0, smoothing=0.0)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ocr_model.parameters()),
    #                            lr=0,
    #                            betas=(0.9, 0.98),
    #                            eps=1e-9)
    optimizer = torch.optim.Adam(ocr_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    model_opt = NoamOpt(d_model, 1, 400, optimizer)

    for epoch in range(nrof_epochs):
        print(f"\nepoch {epoch}")

        print("train...")
        ocr_model.train()
        loss_compute = SimpleLossCompute(ocr_model.generator, criterion, model_opt)
        train_mean_loss = run_epoch(train_loader, ocr_model, loss_compute, device)

        if epoch % 10 == 0:
            print("valid...")
            ocr_model.eval()
            valid_loss_compute = SimpleLossCompute(ocr_model.generator, criterion, None)
            valid_mean_loss = run_epoch(valid_loader, ocr_model, valid_loss_compute, device)
            print(f"valid loss: {valid_mean_loss}")

    # save model
    torch.save(ocr_model.state_dict(), model_save_path)

    # 训练结束，使用贪心的解码方式推理训练集和验证集，统计正确率
    ocr_model.eval()
    print("\n------------------------------------------------")
    print("greedy decode trainset")
    total_img_num = 0
    total_correct_num = 0
    for batch_idx, batch in enumerate(train_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)

        bs = img_input.shape[0]
        for i in range(bs):
            cur_img_input = img_input[i].unsqueeze(0)
            cur_encode_mask = encode_mask[i].unsqueeze(0)
            cur_decode_out = decode_out[i]

            pred_result = greedy_decode(ocr_model, cur_img_input, cur_encode_mask, max_len=sequence_len, start_symbol=1, end_symbol=2)
            pred_result = pred_result.cpu()

            is_correct = judge_is_correct(pred_result, cur_decode_out)
            total_correct_num += is_correct
            total_img_num += 1
            if not is_correct:
                # 预测错误的case进行打印
                print("----")
                print(cur_decode_out)
                print(pred_result)
    total_correct_rate = total_correct_num / total_img_num * 100
    print(f"total correct rate of trainset: {total_correct_rate}%")

    print("\n------------------------------------------------")
    print("greedy decode validset")
    total_img_num = 0
    total_correct_num = 0
    for batch_idx, batch in enumerate(valid_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)

        bs = img_input.shape[0]
        for i in range(bs):
            cur_img_input = img_input[i].unsqueeze(0)
            cur_encode_mask = encode_mask[i].unsqueeze(0)
            cur_decode_out = decode_out[i]

            pred_result = greedy_decode(ocr_model, cur_img_input, cur_encode_mask, max_len=sequence_len, start_symbol=1, end_symbol=2)
            pred_result = pred_result.cpu()

            is_correct = judge_is_correct(pred_result, cur_decode_out)
            total_correct_num += is_correct
            total_img_num += 1
            if not is_correct:
                # 预测错误的case进行打印
                print("----")
                print(cur_decode_out)
                print(pred_result)
    total_correct_rate = total_correct_num / total_img_num * 100
    print(f"total correct rate of validset: {total_correct_rate}%")
