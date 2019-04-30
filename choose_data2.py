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
        user_input, L_input, S_input, pos_item_input, neg_item_input = [], [], [], [], []
        user_input_test, L_input_test, S_input_test, pos_item_input_test, neg_item_input_test = [], [], [], [], []
        for i in range(0, self.user_dict_train.__len__()):
            u = self.user_set[i]
            u_id = i
            user_info = self.user_dict_train[u]
            for time in user_info.keys():
                L = []
                S = []
                j = 0
                for item in user_info.items():  # 分别存入用户长期和短期项目集
                    if int(item[0]) >= int(time):
                        j = self.get_single_item_id(item[1])
                    elif int(item[0]) > int(time) - timeStep:
                        S.append(self.get_single_item_id(item[1]))
                    else:
                        L.append(self.get_single_item_id(item[1]))
                if L.__len__() == 0 and S.__len__() == 0:
                    continue
                k = self.get_unob(L, S)
                if int(time) > currentTime - timeStep and np.random.uniform(0, 1) < 0.25:
                    user_input_test.append(u_id)
                    L_input_test.append(L)
                    S_input_test.append(S)
                    pos_item_input_test.append(j)
                    neg_item_input_test.append(k)
                else:
                    user_input.append(u_id)
                    L_input.append(L)
                    S_input.append(S)
                    pos_item_input.append(j)
                    neg_item_input.append(k)

        # 填充L_input
        L_max = 0
        for i in range(0, L_input.__len__()):
            if L_input[i].__len__() > L_max:
                L_max = L_input[i].__len__()
        for i in range(0, L_input.__len__()):
            while L_input[i].__len__() < L_max:
                L_input[i].append(-1)

        # 填充S_input
        S_max = 0
        for i in range(0, S_input.__len__()):
            if S_input[i].__len__() > S_max:
                S_max = S_input[i].__len__()
        for i in range(0, S_input.__len__()):
            while S_input[i].__len__() < S_max:
                S_input[i].append(-1)

        # 填充L_input_test
        L_max_test = 0
        for i in range(0, L_input_test.__len__()):
            if L_input_test[i].__len__() > L_max_test:
                L_max_test = L_input_test[i].__len__()
        for i in range(0, L_input_test.__len__()):
            while L_input_test[i].__len__() < L_max_test:
                L_input_test[i].append(-1)

        # 填充S_input
        S_max_test = 0
        for i in range(0, S_input_test.__len__()):
            if S_input_test[i].__len__() > S_max_test:
                S_max_test = S_input_test[i].__len__()
        for i in range(0, S_input_test.__len__()):
            while S_input_test[i].__len__() < S_max_test:
                S_input_test[i].append(-1)
        # print(user_input.__len__())
        # tensorU = torch.LongTensor(L_input)
        # print('tensorU')
        # print(tensorU)
        # print(tensorU.size())
        print(user_input.__len__())
        print(user_input_test.__len__())
        return user_input, L_input, S_input, pos_item_input, neg_item_input,\
               user_input_test, L_input_test, S_input_test, pos_item_input_test, neg_item_input_test

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
user_input, L_input, S_input, pos_item_input, neg_item_input, user_input_test, L_input_test, S_input_test, pos_item_input_test, neg_item_input_test = data.get_ob()
write_file('data/user_input_train.txt', user_input)
write_file('data/L_input_train.txt', L_input)
write_file('data/S_input_train.txt', S_input)
write_file('data/pos_item_input_train.txt', pos_item_input)
write_file('data/neg_item_input_train.txt', neg_item_input)

write_file('data/user_input_test.txt', user_input_test)
write_file('data/L_input_test.txt', L_input_test)
write_file('data/S_input_test.txt', S_input_test)
write_file('data/pos_item_input_test.txt', pos_item_input_test)
write_file('data/neg_item_input_test.txt', neg_item_input_test)

with open(r'data\user_input_train.txt', 'r', encoding='UTF-8') as testF:
    test_list = eval(testF.read())
with open(r'data\user_input_test.txt', 'r', encoding='UTF-8') as testF:
    test_list2 = eval(testF.read())
print("train_list.__len__()", test_list.__len__())
print(test_list)
print("test_list.__len__()", test_list2.__len__())
print(test_list2)
