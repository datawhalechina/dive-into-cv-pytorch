# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:45:47 2019

封装的一个模型存储器

用于保存模型
同时确保训练过程中只保存最近产生的几个模型
避免过多的硬盘空间占用

@author: zyb_as
"""
import os
import torch


class ModelSaver:
    def __init__(self, max_save_num = 5):
        self.max_save_num = max_save_num
        self.model_path_list = []
        self.oldest_idx = -1
        self.newest_idx = -1

    def save_new_model(self, net, model_save_path):
        save_model_num = self.newest_idx - self.oldest_idx
        if save_model_num == self.max_save_num:
            # model save num == threshold, delete the oldest one and save new model

            # delete the oldest model
            self.oldest_idx += 1  
            oldest_model_path = self.model_path_list[self.oldest_idx]
            os.remove(oldest_model_path)

            # save new model
            torch.save(net.state_dict(), model_save_path) 
            self.model_path_list.append(model_save_path)
            self.newest_idx += 1
        else:
            # model save num < threshold, save model directly
            torch.save(net.state_dict(), model_save_path) 
            self.model_path_list.append(model_save_path)
            self.newest_idx += 1
