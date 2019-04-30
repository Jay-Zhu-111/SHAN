import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class SHAN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, drop_ratio):
        super(SHAN, self).__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.embedding_dim = embedding_dim
        # self.predictlayer = PredictLayer(embedding_dim, drop_ratio)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w_range = np.sqrt(3 / embedding_dim)
                nn.init.uniform_(m.weight, -w_range, w_range)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean = 0, std = 0.01)

    def forward(self, user_inputs, L_inputs, S_inputs, item_inputs):
        # item_embeds_full = self.itemembeds(Variable(torch.LongTensor(item_inputs), requires_grad=False))
        re = torch.Tensor()
        re_user = torch.Tensor()
        re_item = torch.Tensor()
        for user, L_, S_, item in zip(user_inputs, L_inputs, S_inputs, item_inputs):
            # print("user",user)
            # print("L",L_)
            # print("S",S_)
            # print("item",item)
            L = []
            for i in range(0, L_.__len__()):
                if L_[i] != -1:
                    L.append(L_[i])
            # print("L",L)
            S = []
            for i in range(0, S_.__len__()):
                if S_[i] != -1:
                    S.append(S_[i])
            # if L.__len__() == 0 and S.__len__() == 0:
            #     print(L)
            #     print(S)
            # print("S",S)
            item = [item]
            # user = [user]
            # item_embed = self.itemembeds(Variable(torch.LongTensor(item)))
            # print(user__)
            # print(user)
            # user_embed = self.userembeds(torch.LongTensor(user))
            # print(list(self.userembeds.parameters()))
            # if L.__len__() == 0 or S.__len__() == 0:
            #     print(L)
            #     print(S)
            if L.__len__() != 0:
                # 第一层注意力网络
                # 用户嵌入 L * K
                user_numb1 = []
                for _ in L:
                    user_numb1.append(user)
                user_embed1 = self.userembeds(Variable(torch.LongTensor(user_numb1)))
                # print(user_embed1)
                # 长期项目集嵌入 L * K
                L_embed = self.itemembeds(Variable(torch.LongTensor(L)))
                # 连接用户和项目嵌入 L * 2K
                user_L_embed = torch.cat((user_embed1, L_embed), dim=1)
                # print(user_L_embed)
                # 权重 L * 1
                at_wt1 = self.attention(user_L_embed)
                # print("at_wt", at_wt1)
                # 用户长期表示 1 * K
                u_long = torch.matmul(at_wt1, L_embed)

            if S.__len__() != 0:
                if L.__len__() != 0:
                    # 第二层注意力网络
                    # 用户嵌入 (S + 1) * K
                    user_numb2 = [user]
                    for _ in S:
                        user_numb2.append(user)
                    user_embed2 = self.userembeds(Variable(torch.LongTensor(user_numb2)))
                    # 短期项目集嵌入 S * K
                    S_embed = self.itemembeds(Variable(torch.LongTensor(S)))
                    # 连接用户长期表示和短期项目集嵌入 (S + 1) * K
                    u_long_S_embed = torch.cat((u_long, S_embed), dim=0)
                    # 连接用户和项目嵌入 (S + 1) * 2K
                    user_u_long_S_embed = torch.cat((user_embed2, u_long_S_embed), dim=1)
                    # 权重 (S + 1) * 1
                    at_wt2 = self.attention(user_u_long_S_embed)
                    # 用户混合表示 1 * K
                    user_hybrid = torch.matmul(at_wt2, u_long_S_embed)
                else:
                    # 第二层注意力网络
                    # 用户嵌入 S * K
                    user_numb2 = []
                    for _ in S:
                        user_numb2.append(user)
                    user_embed2 = self.userembeds(Variable(torch.LongTensor(user_numb2)))
                    # 短期项目集嵌入 S * K
                    S_embed = self.itemembeds(Variable(torch.LongTensor(S)))
                    # 连接用户和项目嵌入 S * 2K
                    user_S_embed = torch.cat((user_embed2, S_embed), dim=1)
                    # 权重 S * 1
                    at_wt2 = self.attention(user_S_embed)
                    # 用户混合表示 1 * K
                    user_hybrid = torch.matmul(at_wt2, S_embed)
            else:
                user_hybrid = u_long
            user_hybrid = torch.reshape(user_hybrid, (1, 1, self.embedding_dim))
            if re_user.size(0) == 0:
                re_user = user_hybrid
            else:
                re_user = torch.cat((re_user, user_hybrid), 0)
            # 得到项目嵌入向量
            item_embed = self.itemembeds(Variable(torch.LongTensor(item)))
            item_embed = torch.reshape(item_embed, (1, 1, self.embedding_dim))
            if re_item.size(0) == 0:
                re_item = item_embed
            else:
                re_item = torch.cat((re_item, item_embed), 0)
            # 得到项目评分
            # score = torch.matmul(user_hybrid, torch.transpose(item_embed, 0 ,1))
            # if re.size(0) == 0:
            #     re = score
            # else:
            #     re = torch.cat((re, score), 0)
            # print("re", re)
        # print(re_user.size())
        # print(torch.transpose(re_item, 1, 2).size())
        re = torch.bmm(re_user, torch.transpose(re_item, 1, 2))
        return re


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)
        # torch.nn.init.normal(self.userEmbedding.weight)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


# class GroupEmbeddingLayer(nn.Module):
#     def __init__(self, number_group, embedding_dim):
#         super(GroupEmbeddingLayer, self).__init__()
#         self.groupEmbedding = nn.Embedding(number_group, embedding_dim)
#
#     def forward(self, num_group):
#         group_embeds = self.groupEmbedding(num_group)
#         return group_embeds
#
#
class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight


# class PredictLayer(nn.Module):
#     def __init__(self, embedding_dim, drop_ratio=0):
#         super(PredictLayer, self).__init__()
#         self.linear = nn.Sequential(
#             nn.Linear(embedding_dim, 8),
#             nn.ReLU(),
#             nn.Dropout(drop_ratio),
#             nn.Linear(8, 1)
#         )
#
#     def forward(self, x):
#         out = self.linear(x)
#         return out

