import Const
import random
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

timeStep = Const.time_step
currentTime = Const.current_time


class Data:
    def __init__(self):
        with open(r'data\user_dict.txt', 'r', encoding='UTF-8') as user_dict_train_f:
            self.user_dict_train = eval(user_dict_train_f.read())

        with open(r'data\item_set.txt', 'r', encoding='UTF-8') as item_set_f:
            self.item_set = eval(item_set_f.read())

        with open(r'data\user_set.txt', 'r', encoding='UTF-8') as user_set_f:
            self.user_set = eval(user_set_f.read())

        with open(r'C:\Users\CrazyCat\Desktop\LA-Venues.txt', 'r', encoding='UTF-8') as f:
            self.all_item_list = f.readlines()

        with open('data/item_dict.txt', 'r', encoding='UTF-8') as f:
            self.item_dict = eval(f.read())

    # 总用户数
    def get_user_size(self):
        return self.user_set.__len__()

    # 总项目数
    def get_item_size(self):
        return self.item_set.__len__()

    # 返回物品
    def get_single_item(self, item_id):
        return self.item_set[item_id]

    # 返回物品序号
    def get_single_item_id(self, item):
        re = 0
        for i in range(0, self.item_set.__len__()):
            if item == self.item_set[i]:
                re = i
                break
        return re

    # 返回物品序号集
    def get_item_id(self, input_set):
        item_list = []
        for item in input_set:
            for i in range(0, self.item_set.__len__()):
                if item == self.item_set[i]:
                    item_list.append(i)
                    break
        return item_list

    # 获取一个观察样本
    def get_ob(self):
        L_all, S_all = [], []
        for i in range(0, self.user_dict_train.__len__()):
            L = []
            S = []
            u = self.user_set[i]
            user_info = self.user_dict_train[u]
            for item in user_info.items():  # 分别存入用户长期和短期项目集
                if int(item[0]) > currentTime - timeStep:
                    S.append(self.get_single_item_id(item[1]))
                else:
                    L.append(self.get_single_item_id(item[1]))
            L_all.append(L)
            S_all.append(S)
        return L_all, S_all

    # 随机获取一个负样本
    def get_unob(self, L, S):
        ob_item = []
        for item in L:
            ob_item.append(item)
        for item in S:
            ob_item.append(item)
        while True:
            k = random.randint(0, self.item_set.__len__() - 1)
            if k not in ob_item:
                break
        return k

    def get_user_set(self):
        return self.user_set

    def get_item_set(self):
        return self.item_set

    def get_item_dict(self):
        return self.item_dict


def write_file(filename, data):
    file = open(filename, 'w')
    file.write(str(data))
    file.close()


data = Data()
item_set = data.get_item_set()
item_dict = data.get_item_dict()

str_item = ''
file_u = open('data/item.txt', 'w', encoding='UTF-8')

print(item_set.__len__())
for i in range(item_set.__len__()):
    item_info = item_dict[item_set[i]]
    str_item += str(i)
    str_item += ';'
    str_item += item_info[0]
    str_item += ';'
    str_item += item_info[1]
    str_item += ';'
    str_item += item_info[2]
    str_item += '\n'

file_u.write(str_item)
file_u.close()
