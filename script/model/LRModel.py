from model import Model


from sklearn.linear_model import LogisticRegression


class LRModel(Model):
    def __init__(self):
        super(LRModel, self).__init__()
        self.model = LogisticRegression()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict_proba(self, x_test):
        # 取出标签为1所属列
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_test)[:, pos_index]

    def predict(self, x_test):
        return self.model.predict(x_test)