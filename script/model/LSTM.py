import math

from torch import nn
import torch

import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score, matthews_corrcoef


class LSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=100, output_size=2, num_layer=2):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Dropout(p=0.2))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.input_size))
        return zeros

    def forward(self, batch_data):
        # (batch_size , seq_length , hidden_size)
        batch_data, _ = self.layer1(batch_data)
        # 转换维度
        # (batch_size , hidden_size, seq_length)
        batch_data = torch.transpose(batch_data, 1, 2)
        # 降维，去除seq_length
        # (batch_size , hidden_size)
        x = F.max_pool1d(batch_data, batch_data.size(2)).squeeze(2)
        x = self.linear(x)
        return x

    def train(self, train_data: list, train_label: list, batch_size= 64, EPOCHS=15):
        count = len(train_data)
        train_data = np.array(train_data)
        train_data = torch.from_numpy(train_data)
        train_data = train_data.view(count, 1, -1)
        train_data = train_data.to(torch.float32)
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
    model = LSTM(2, 4, 2, 2)

    x = torch.randn(2, 3, 2)
    x = [[[1, 1],
          [2, 2],
          [3, 3]],
         [[2, 1],
          [3, 4],
          [1, 3]]]
    x = torch.FloatTensor(x)
    print(x.shape)
    print(model)
    output = model(x)
    print(output)
    _, predicted = torch.max(output, 1)
    print(predicted)
