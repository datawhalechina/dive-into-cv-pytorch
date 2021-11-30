# -*- coding: utf-8 -*-
"""
分析本节所要用的识别数据集

本节的OCR小实验使用的数据集是 ICDAR 2015 Incidental Scene Text Task4.3 word Recognition
本脚本会对这个数据集进行一些简单的分析，并统计生成一些后续需要使用的文件
"""
import os
from PIL import Image
import tqdm

from tensorbay import GAS
from tensorbay.dataset import Dataset


def read_gas_image(data):
    with data.open() as fp:
        image = Image.open(fp)
    return image


def statistics_max_len_label(segment):
    """
    统计标签中最长的label所包含的字符数
    """
    max_len = -1
    for data in segment:
        lbl_str = data.label.classification.category  # 获取标签
        lbl_len = len(lbl_str)
        max_len = max_len if max_len > lbl_len else lbl_len
    return max_len


def statistics_label_cnt(segment, lbl_cnt_map):
    """
    统计标签文件中都包含哪些label以及各自出现的次数
    """
    for data in segment:
        lbl_str = data.label.classification.category  # 获取标签
        for lbl in lbl_str:
                if lbl not in lbl_cnt_map.keys():
                    lbl_cnt_map[lbl] = 1
                else:
                    lbl_cnt_map[lbl] += 1


def load_lbl2id_map(lbl2id_map_path):
    """
    读取label-id映射关系记录文件
    """
    lbl2id_map = dict()
    id2lbl_map = dict()
    with open(lbl2id_map_path, 'r') as reader:
        for line in reader:
            items = line.rstrip().split('\t')
            label = items[0]
            cur_id = int(items[1])
            lbl2id_map[label] = cur_id
            id2lbl_map[cur_id] = label
    return lbl2id_map, id2lbl_map


if __name__ == "__main__":

    # GAS凭证
    KEY = 'Accesskey-fd26cc098604c68a99d3bf7f87cd480a'
    gas = GAS(KEY)
    # Get a dataset. 在线获取数据集
    dataset = Dataset("ICDAR2015", gas)
    dataset.enable_cache('./data')  # 数据缓存地址

    # 获取训练集和验证集
    train_segment = dataset["train"]
    valid_segment = dataset['valid']

    # 数据集标签映射字段存储目录
    base_data_dir = './'
    lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')

    # 统计数据集中出现的所有的label中包含字符最多的有多少字符
    train_max_label_len = statistics_max_len_label(train_segment)
    valid_max_label_len = statistics_max_len_label(valid_segment)
    max_label_len = max(train_max_label_len, valid_max_label_len)
    print(f"数据集中包含字符最多的label长度为{max_label_len}")

    # 统计数据集中出现的所有的符号
    lbl_cnt_map = dict()
    statistics_label_cnt(train_segment, lbl_cnt_map)
    print("训练集中出现的label")
    print(lbl_cnt_map)
    statistics_label_cnt(valid_segment, lbl_cnt_map)
    print("训练集+验证集中出现的label")
    print(lbl_cnt_map)

    # 构造 label - id 之间的映射
    print("\n构造 label - id 之间的映射")
    lbl2id_map = dict()
    # 初始化两个特殊字符
    lbl2id_map['☯'] = 0    # padding标识符
    lbl2id_map['■'] = 1    # 句子起始符
    lbl2id_map['□'] = 2    # 句子结束符
    # 生成其余label的id映射关系
    cur_id = 3
    for lbl in lbl_cnt_map.keys():
        lbl2id_map[lbl] = cur_id
        cur_id += 1
    # 保存 label - id 之间的映射
    with open(lbl2id_map_path, 'w', encoding='utf-8') as writer:
        for lbl in lbl2id_map.keys():
            cur_id = lbl2id_map[lbl]
            print(lbl, cur_id)
            line = lbl + '\t' + str(cur_id) + '\n'
            writer.write(line)

    # 分析数据集图片尺寸
    print("分析数据集图片尺寸")
    min_h = 1e10
    min_w = 1e10
    max_h = -1
    max_w = -1
    min_ratio = 1e10
    max_ratio = 0
    for data in tqdm.tqdm(train_segment):
        img = read_gas_image(data)
        w, h = img.size
        ratio = w / h
        min_h = min_h if min_h <= h else h
        max_h = max_h if max_h >= h else h
        min_w = min_w if min_w <= w else w
        max_w = max_w if max_w >= w else w
        min_ratio = min_ratio if min_ratio <= ratio else ratio
        max_ratio = max_ratio if max_ratio >= ratio else ratio
    print("min_h", min_h)
    print("max_h", max_h)
    print("min_w", min_w)
    print("max_w", max_w)
    print("min_ratio", min_ratio)
    print("max_ratio", max_ratio)
