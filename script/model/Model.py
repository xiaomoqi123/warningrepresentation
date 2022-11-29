import abc

from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score, matthews_corrcoef

import matplotlib.pyplot as plt


class Model:

    @abc.abstractmethod
    def train(self, x_train, y_train):
        """
        模型训练

        :param x_train: 训练数据
        :param y_train: 训练标签
        """
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        """
        返回预测值

        :param x_test: 预测数据
        :return: 预测值
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, x_test):
        """
        返回预测为正值的概率

        :param x_test: 预测数据
        :return: 预测为正概率
        """
        pass

    def predict_multi_proba(self, x_test):
        proba = self.model.predict_proba(x_test)
        return proba

    def accuracy_score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return accuracy_score(y_test, y_predict)

    def recall_score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return recall_score(y_test, y_predict)

    def precision_score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return precision_score(y_test, y_predict)

    def get_result_statistics(self, x_test, y_test):
        predictions = self.predict(x_test)
        proba = self.predict_proba(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        print("TN, FP, FN, TP: ", (tn, fp, fn, tp))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        auc = roc_auc_score(y_test, proba)
        mcc = matthews_corrcoef(y_test, predictions)

        PF = fp / (fp + tn)
        f1 = 2 * recall * precision / (recall + precision)
        g_measure = 2 * recall * (1 - PF) / (recall + 1 - PF)
        return tp, tn, fp, fn, precision, recall, f1, acc, auc, mcc, g_measure

    def get_multi_classification_result_statistics(self, x_test, y_test, n_class):
        predictions = self.predict(x_test)
        cm = confusion_matrix(y_test, predictions, labels=n_class)
        plt.matshow(cm, cmap=plt.cm.gray)

        acc = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='macro')
        precision = precision_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')
        mcc = matthews_corrcoef(y_test, predictions)
        auc = 0
        try:
            auc = roc_auc_score(y_test, self.predict_multi_proba(x_test), multi_class='ovo')
        except:
            auc = 0
        print("acc, recall, precision, f1, mcc, auc: ", (acc, recall, precision, f1, mcc, auc))
        return plt, acc, recall, precision, f1, mcc, auc

        # cm = cm.astype(np.float32)
        #         # FP = cm.sum(axis=0) - np.diag(cm)
        #         # FN = cm.sum(axis=1) - np.diag(cm)
        #         # TP = np.diag(cm)
        #         # TN = cm.sum() - (FP + FN + TP)
        #         # recall = TP / (TP + FN)
        #         # precision = TP / (TP + FP)
        #         # acc = (TP + TN) / (TP + TN + FP + FN)
        #         # mcc = matthews_corrcoef(y_test, predictions)
        #         #
        #         # PF = FP / (FP + TN)
        #         # f1 = 2 * recall * precision / (recall + precision)
        #         # g_measure = 2 * recall * (1 - PF) / (recall + 1 - PF)
        #         # print("{},{},{},{},{},{},{},{},{},{}".format(TP, TN, FP, FN, precision, recall, f1, acc, mcc, g_measure))
        # return TP, TN, FP, FN, precision, recall, f1, acc, mcc, g_measure
