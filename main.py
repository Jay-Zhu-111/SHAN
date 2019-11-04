from get_data import Data
from _model import SHAN
import Const
import torch
import torch.optim as optim
from time import time
from helper import Helper

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

        # if torch.cuda.is_available():
        #     user_input = user_input.cuda()
        #     L_input = L_input.cuda()
        #     S_input = S_input.cuda()
        #     pos_item_input = pos_item_input.cuda()
        #     neg_item_input = neg_item_input.cuda()
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
        FT = LT.float()
        # crit=torch.nn.MSELoss()
        # loss=crit(torch.sigmoid(FT),torch.tensor(1.0))
        loss = torch.mean(torch.neg(torch.log((torch.sigmoid(FT)))))
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
    count = [0.0, 0.0, 0.0, 0.0]
    for num in hits:
        for i in range(count.__len__()):
            if num[i] == 1:
                count[i] += 1
    Recall = []
    for i in range(count.__len__()):
        Recall.append(count[i] / hits.__len__())

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
    # shan.load_state_dict(torch.load('SHAN2_dict.pkl'))
    # print(shan)

    if torch.cuda.is_available():
        print("using cuda")
        shan.cuda()

    lr_flag = True
    pre_mean_loss = 999
    lr = Const.lr
    for i in range(0, epoch):
        shan.train()
        # 开始训练时间
        t1 = time()
        # if lr_flag:
        #     lr *= 1.1
        #     mean_loss = training(shan, data.get_dataloader(batch_size), i, lr)
        # else:
        #     lr *= 0.5
        #     mean_loss = training(shan, data.get_dataloader(batch_size), i, lr)
        # if mean_loss < pre_mean_loss:
        #     lr_flag = True
        # else:
        #     lr_flag = False
        # pre_mean_loss = mean_loss
        mean_loss = training(shan, data.get_dataloader(batch_size), i, lr)
        print("learning rate is: ", lr)
        print("Training time is: [%.5f s]" % (time() - t1))

        # evaluating
        t2 = time()
        Recall, AUC = evaluation(shan, h)
        print("Recall@5", Recall[0])
        print("Recall@10", Recall[1])
        print("Recall@15", Recall[2])
        print("Recall@20", Recall[3])
        print("AUC", AUC)
        print("Evalulating time is: [%.1f s]" % (time() - t2))
        print("\n")

        torch.save(shan.state_dict(), 'SHAN5_dict.pkl')
        print("______________save______________")

    # helper.draw_loss()
    print("Done!")
