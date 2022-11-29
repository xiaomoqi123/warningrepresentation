import math

import torch.nn as nn
import torch


from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, matthews_corrcoef

import numpy as np


class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        # input_channels:输入通道，文本应该为1，图片可能有3通道，即为RGB
        # out_channels:输出通道，即为filter_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, input_size), padding=0),
            nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(32, 16)
        # self.fc = nn.Linear(16 , 2)

        # linear nn.Softmax(dim=None) 归一化
        self.fc = nn.Sequential(nn.Linear(16, 2), nn.Dropout(p=0.2))

    def forward(self, x):

        # print(out.shape)

        x = self.conv1(x)
        print(x.shape)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = x.squeeze(-1)
        # print(out.shape)
        # 池化
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # print(out.shape)
        out = self.linear1(x)
        return self.fc(out)

    def train(self, train_data: list, train_label: list, batch_size=64, EPOCHS=15):
        count = len(train_data)
        train_data = np.array(train_data)
        train_data = torch.from_numpy(train_data)
        train_data = train_data.view(count, 1, -1)
        train_data = train_data.to(torch.float32)
        train_data = train_data.unsqueeze(1)
        parameters = self.parameters()
        optimizer = torch.optim.Adadelta(parameters)
        loss_function = torch.nn.CrossEntropyLoss()
        for epcho in range(EPOCHS):
            i = 0
            while i < len(train_data):
                cur_train_features = train_data[i:i + batch_size]
                cur_train_labels = train_label[i:i + batch_size]
                optimizer.zero_grad()
                output = self(cur_train_features)
                # 反向传播，获得最佳模型
                loss = loss_function(output, Variable(cur_train_labels))
                print('epcho:{} , loss:{}'.format(epcho, loss))
                loss.backward()
                optimizer.step()
                i += batch_size

    def get_result_statistics(self, x_test, y_test, batch_size=64):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        prob_all = []
        label_all = []
        predict_all = []

        count = len(x_test)
        x_test = np.array(x_test)
        x_test = torch.from_numpy(x_test)
        x_test = x_test.view(count, 1, -1)
        x_test = x_test.to(torch.float32)
        x_test = x_test.unsqueeze(1)

        i = 0
        while i < len(x_test):
            cur_test_features = x_test[i: i + batch_size]
            cur_test_labels = y_test[i: i + batch_size]
            output = self(cur_test_features)
            cur_test_labels = torch.LongTensor(cur_test_labels)
            _, predicted = torch.max(output, 1)

            TP += ((predicted == 1) & (cur_test_labels.data == 1)).cpu().sum()
            # TN    predict 和 label 同时为0
            TN += ((predicted == 0) & (cur_test_labels.data == 0)).cpu().sum()
            # FN    predict 0 label 1
            FN += ((predicted == 0) & (cur_test_labels.data == 1)).cpu().sum()
            # FP    predict 1 label 0
            FP += ((predicted == 1) & (cur_test_labels.data == 0)).cpu().sum()

            prob_all.extend(
                output[:, 1].cpu().detach().numpy())
            label_all.extend(cur_test_labels)
            predict_all.extend(predicted)
            i += batch_size

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * recall * precision / (recall + precision)
        acc = (TP + TN) / (TP + TN + FP + FN)
        auc = roc_auc_score(label_all, prob_all)
        mcc = matthews_corrcoef(label_all, predict_all)
        g_measure = math.sqrt(precision * recall)
        print("TP:{} , TN:{} , FP:{} , FN:{}".format(TP, TN, FP, FN))
        print("Precision:{} , Recall:{} , F1:{} , ACC:{} , AUC:{} , MCC:{} , G_Measure:{}".format(precision, recall, f1,
                                                                                                  acc, auc, mcc,
                                                                                                  g_measure))
        return TP.item(), TN.item(), FP.item(), FN.item(), \
               precision.item(), recall.item(), f1.item(), acc.item(), auc, \
               mcc, g_measure



if __name__ == '__main__':
    model = CNN()
    t1 = torch.rand(2, 1, 50, 128)
    print(model(t1))
