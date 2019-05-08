from get_data import Data
from _model import SHAN
from matplotlib.pyplot import *
import Const
import torch
import torch.optim as optim
from time import time
import numpy as np
import helper
from helper import Helper
from torch.autograd import Variable

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# train the model
def training(model, train_loader, epoch_id, lr):
    # user trainning
    # learning_rates = Const.lr
    # learning rate decay
    # lr = learning_rates
    # if epoch_id >= 15 and epoch_id < 25:
    #     lr = learning_rates[1]
    # elif epoch_id >=20:
    #     lr = learning_rates[2]
    # # lr decay
    # if epoch_id % 5 == 0:
    #     lr /= 2

    # optimizer 正则化参数？？
    optimizer = optim.Adam(model.parameters(), lr)

    losses = []
    for user_input, L_input, S_input, pos_item_input, neg_item_input in train_loader:
        # # Data Load
        # user_input = user
        # L_input = L_S_pi_ni[:, 0]
        # S_input = L_S_pi_ni[:, 1]
        # pos_item_input = L_S_pi_ni[:, 2]
        # neg_item_input = L_S_pi_ni[:, 3]
        # Data Clean 删除冗余的-1
        # L_input = torch.Tensor(helper.clean_data(L_input))
        # S_input = torch.Tensor(helper.clean_data(S_input))

        # Forward
        pos_prediction = model(user_input, L_input, S_input, pos_item_input)
        neg_prediction = model(user_input, L_input, S_input, neg_item_input)

        # Zero_grad
        model.zero_grad()

        # Loss
        # LT = pos_prediction - neg_prediction
        # FT = LT.float()
        # loss = Variable(torch.mean(torch.neg(torch.log((torch.sigmoid(FT))))),
        #                 requires_grad=True)
        LT = pos_prediction - neg_prediction
        # print(pos_prediction, neg_prediction)
        FT = LT.float()
        # crit=torch.nn.MSELoss()
        # loss=crit(torch.sigmoid(FT),torch.tensor(1.0))
        loss = torch.mean(torch.neg(torch.log((torch.sigmoid(FT)))))
        # print(loss.requires_grad)
        # print("epoch_id", epoch_id)
        # print("loss", loss)
        # helper.save_loss(loss)
        # record loss history
        losses.append(float(loss))

        # Backward
        loss.backward()
        optimizer.step()

    mean_loss = 0
    for loss in losses:
        mean_loss += loss
    mean_loss /= losses.__len__()
    print("epoch_id", epoch_id)
    print("mean_loss", mean_loss)
    # print('Iteration %d, loss is [%.4f ]' % (epoch_id, losses ))
    return mean_loss


def evaluation(model, Helper):
    model.eval()
    (hits, AUCs) = Helper.evaluate_model(model)

    # Recall
    count = 0.0
    for num in hits:
        if num == 1:
            count = count + 1
    Recall = count / hits.__len__()

    # AUC
    count = 0.0
    for num in AUCs:
        count = count + num
    AUC = count / AUCs.__len__()

    return Recall, AUC


if __name__ == '__main__':
    embedding_size = Const.embedding_size
    drop_ratio = Const.drop_ratio
    epoch = Const.epoch
    batch_size = Const.batch_size

    data = Data()
    h = Helper()
    num_users = data.get_user_size()
    num_items = data.get_item_size()
    shan = SHAN(num_users, num_items, embedding_size, drop_ratio)
    # print(shan)

    lr_flag = True
    pre_mean_loss = 999
    lr = Const.lr
    for i in range(0, epoch):
        shan.train()
        # 开始训练时间
        t1 = time()
        if lr_flag:
            lr *= 1.1
            mean_loss = training(shan, data.get_dataloader(batch_size), i, lr)
        else:
            lr *= 0.5
            mean_loss = training(shan, data.get_dataloader(batch_size), i, lr)
        if mean_loss < pre_mean_loss:
            lr_flag = True
        else:
            lr_flag = False
        pre_mean_loss = mean_loss
        print("learning rate is: ", lr)
        print("Training time is: [%.1f s]" % (time() - t1))

        # evaluating
        t2 = time()
        Recall, AUC = evaluation(shan, h)
        print("Recall", Recall)
        print("AUC", AUC)
        print("Evalulating time is: [%.1f s]" % (time() - t2))
        print("\n")

    # helper.draw_loss()
    torch.save(shan, 'SHAN.pkl')
    print("Done!")
