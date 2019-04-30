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

        # # user = input("输入用户ID:")
        # self.u = '6756'
        # for i in range(0, self.user_set.__len__()):
        #     if self.user_set[i] == self.u:
        #         self.u_id = i
        #         break
        #
        # self.user_info = self.user_dict[self.u]
        #
        # self.S = []
        # self.L = []
        # for item in self.user_info.items():  # 分别存入用户长期和短期项目集
        #     if int(item[0]) > currentTime - timeStep:
        #         self.S.append(item[1])
        #     else:
        #         self.L.append(item[1])

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

    # # 返回用户ID
    # def get_u(self):
    #     return self.u_id
    #
    # # 返回长期项目集
    # def get_L(self):
    #     return self.L
    #
    # # 返回短期项目集
    # def get_S(self):
    #     return self.S

    # # 获取观察样本集
    # def get_ob_set(self):
    #     ob_set = []
    #     for item in self.user_info.items():
    #         (u, L, S, j) = self.get_ob(int(item[0]))
    #         ob_set.append((u, L, S, j))
    #     return ob_set

    # 获取一个观察样本
    def get_ob(self):
        L, S = [], []
        for i in range(0, self.user_dict_train.__len__()):
            u = self.user_set[i]
            u_id = i
            user_info = self.user_dict_train[u]
            for item in user_info.items():  # 分别存入用户长期和短期项目集
                if int(item[0]) > currentTime - timeStep:
                    S.append((u, self.get_single_item_id(item[1])))
                else:
                    L.append((u, self.get_single_item_id(item[1])))
            if L.__len__() == 0 and S.__len__() == 0:
                continue
        return L, S

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

    # 返回训练dataloader
    def get_dataloader(self, batch_size, is_training):
        user, L, S, j, k = self.get_ob(is_training)
        train_data = TensorDataset(
            torch.LongTensor(user),
            torch.LongTensor(L),
            torch.LongTensor(S),
            torch.LongTensor(j),
            torch.LongTensor(k)
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_loader

    # # 返回dataframe
    # def get_dataframe(self):
    #     data = pd.DataFrame.from_dict(self.user_info, orient='index')
    #     data.rename(columns = {'user', 'item'})
    #     return data
    #
    # # 返回dataset
    # def read_data(self, batch_size, is_training):
    #     user_dict = {}
    #     user_dict['time'] = np.array( list(self.user_info.keys()) )
    #     user_dict['item'] = np.array(list(self.user_info.values()))
    #     dataset = tf.data.Dataset.from_tensor_slices(user_dict)
    #     if is_training:
    #         dataset = dataset.shuffle(10000)
    #     dataset = dataset.batch(batch_size)
    #
    #     return dataset


def write_file(filename, data):
    file = open(filename, 'w')
    file.write(str(data))
    file.close()


data = Data()
L, S = data.get_ob()
print(L)
print(L.__len__())
print(S)
print(S.__len__())
str_L = ''
file_L = open('data/user_L.txt', 'w')
for i in range(0, L.__len__()):
    (user, item) = L[i]
    str_L += user
    str_L += ' '
    str_L += str(item)
    str_L += '\n'
file_L.write(str_L)
file_L.close()

str_S = ''
file_S = open('data/user_S.txt', 'w')
for i in range(0, S.__len__()):
    (user, item) = S[i]
    str_S += user
    str_S += ' '
    str_S += str(item)
    str_S += '\n'
file_S.write(str_S)
file_S.close()

str_u = ''
i = 2
file_u = open('data/user_password.txt', 'w')
user_set = data.get_user_set()
for u in user_set:
    str_u += str(i)
    i = i + 1
    str_u += ' '
    str_u += u
    str_u += ' '
    str_u += '123qwe'
    str_u += ' '
    str_u += str(1)
    str_u += ' '
    str_u += str(0)
    str_u += '\n'

file_u.write(str_u)
file_u.close()