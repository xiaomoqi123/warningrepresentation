from sklearn.ensemble import RandomForestClassifier

from model import Model


class RFModel(Model):

    def __init__(self):
        super(RFModel, self).__init__()
        self.model = RandomForestClassifier(oob_score=True, n_estimators=40)
        # self.model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=4)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict_proba(self, x_test):
        # 取出标签为1所属列
        pos_index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x_test)[:, pos_index]

    def predict(self, x_test):
        return self.model.predict(x_test)

    def score(self , x_test , y_test):
        return self.model.score(x_test , y_test)

if __name__ == '__main__':
    X = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    y = [0, 0, 0, 1, 1, 1, 1 , 1 , 1]
    neigh = RFModel()
    neigh.train(X , y)
    x_test = [[0.5] , [1.5] , [2.5] , [3.5] , [4.5] , [5.5]]
    print(neigh.predict(x_test))