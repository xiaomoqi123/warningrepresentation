from model import KNNModel
from model import RFModel
from model import SvmModel
from model import DecisionTreeModel
import pandas as pd

def train_with_astnn():
    print('----train-with-astnn----')
    train_dataFrame = pd.read_pickle('../prepareData/service/dev/bestFeatures.pkl')
    x_train = train_dataFrame['astnn'].tolist()
    y_train = train_dataFrame['label'].tolist()
    y_train = [1 if label == 'true' else 0 for label in y_train]


    test_dataFrame = pd.read_pickle('../prepareData/service/test/bestFeatures.pkl')
    x_test = test_dataFrame['astnn'].tolist()
    y_test = test_dataFrame['label'].tolist()

    y_test = [1 if label == 'true' else 0 for label in y_test]


    model = KNNModel()
    model.train(x_train , y_train)
    print(model.accuracy_score(x_test , y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))
    # metrics(y_test , model.predict(x_test))

    model = SvmModel()
    model.train(x_train , y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))
    # metrics(y_test, model.predict(x_test))

    model = RFModel()
    model.train(x_train, y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))
    # metrics(y_test, model.predict(x_test))


    model = DecisionTreeModel()
    model.train(x_train , y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))
    # metrics(y_test, model.predict(x_test))

def train_with_merged_feature():
    print('----train-with-merged-feature----')
    train_dataFrame = pd.read_pickle('../prepareData/service/dev/bestFeatures.pkl')
    x_train = train_dataFrame['astnn'].tolist()

    lineNumFeatures = train_dataFrame['lineNum'].tolist()
    statementNumFeatures = train_dataFrame['statementNum'].tolist()
    branchStatementNumFeatures = train_dataFrame['branchStatementNum'].tolist()
    callNumFeatures = train_dataFrame['callNum'].tolist()
    cycleComplexityFeatures = train_dataFrame['cycleComplexity'].tolist()
    depth = train_dataFrame['depth'].tolist()
    for i in range(len(x_train)):
        x_train[i].append(lineNumFeatures[i])
        x_train[i].append(statementNumFeatures[i])
        x_train[i].append(branchStatementNumFeatures[i])
        x_train[i].append(callNumFeatures[i])
        x_train[i].append(cycleComplexityFeatures[i])
        x_train[i].append(depth[i])


    y_train = train_dataFrame['label'].tolist()
    y_train = [1 if label == 'true' else 0 for label in y_train]

    test_dataFrame = pd.read_pickle('../prepareData/service/test/bestFeatures.pkl')
    x_test = test_dataFrame['astnn'].tolist()

    lineNumFeatures = test_dataFrame['lineNum'].tolist()
    statementNumFeatures = test_dataFrame['statementNum'].tolist()
    branchStatementNumFeatures = test_dataFrame['branchStatementNum'].tolist()
    callNumFeatures = test_dataFrame['callNum'].tolist()
    cycleComplexityFeatures = test_dataFrame['cycleComplexity'].tolist()
    depth = test_dataFrame['depth'].tolist()

    for i in range(len(x_test)):
        x_test[i].append(lineNumFeatures[i])
        x_test[i].append(statementNumFeatures[i])
        x_test[i].append(branchStatementNumFeatures[i])
        x_test[i].append(callNumFeatures[i])
        x_test[i].append(cycleComplexityFeatures[i])
        x_test[i].append(depth[i])

    y_test = test_dataFrame['label'].tolist()

    y_test = [1 if label == 'true' else 0 for label in y_test]

    model = KNNModel()
    model.train(x_train, y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))
    # metrics(y_test , model.predict(x_test))

    # model = SvmModel()
    # model.train(x_train, y_train)
    # print(model.accuracy_score(x_test, y_test))
    # print(model.recall_score(x_test, y_test))
    # print(model.precision_score(x_test, y_test))
    # metrics(y_test, model.predict(x_test))

    model = RFModel()
    model.train(x_train, y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))
    # metrics(y_test, model.predict(x_test))

    model = DecisionTreeModel()
    model.train(x_train, y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))
    # metrics(y_test, model.predict(x_test))

def train_with_metrics():
    print('----train-with-metrics')
    train_dataFrame = pd.read_pickle('../prepareData/service/dev/bestFeatures.pkl')

    x_train = train_dataFrame['lineNum'].tolist()
    x_train = [[i]for i in x_train]
    statementNumFeatures = train_dataFrame['statementNum'].tolist()
    branchStatementNumFeatures = train_dataFrame['branchStatementNum'].tolist()
    callNumFeatures = train_dataFrame['callNum'].tolist()
    cycleComplexityFeatures = train_dataFrame['cycleComplexity'].tolist()
    depth = train_dataFrame['depth'].tolist()
    for i in range(len(x_train)):
        x_train[i].append(statementNumFeatures[i])
        x_train[i].append(branchStatementNumFeatures[i])
        x_train[i].append(callNumFeatures[i])
        x_train[i].append(cycleComplexityFeatures[i])
        x_train[i].append(depth[i])

    y_train = train_dataFrame['label'].tolist()
    y_train = [1 if label == 'true' else 0 for label in y_train]



    test_dataFrame = pd.read_pickle('../prepareData/service/test/bestFeatures.pkl')

    x_test = test_dataFrame['lineNum'].tolist()
    x_test = [[i]for i in x_test]
    statementNumFeatures = test_dataFrame['statementNum'].tolist()
    branchStatementNumFeatures = test_dataFrame['branchStatementNum'].tolist()
    callNumFeatures = test_dataFrame['callNum'].tolist()
    cycleComplexityFeatures = test_dataFrame['cycleComplexity'].tolist()
    depth = test_dataFrame['depth'].tolist()

    for i in range(len(x_test)):
        x_test[i].append(statementNumFeatures[i])
        x_test[i].append(branchStatementNumFeatures[i])
        x_test[i].append(callNumFeatures[i])
        x_test[i].append(cycleComplexityFeatures[i])
        x_test[i].append(depth[i])

    y_test = test_dataFrame['label'].tolist()

    y_test = [1 if label == 'true' else 0 for label in y_test]

    model = KNNModel()
    model.train(x_train, y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))

    model = SvmModel()
    model.train(x_train, y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))

    model = RFModel()
    model.train(x_train, y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))

    model = DecisionTreeModel()
    model.train(x_train, y_train)
    print(model.accuracy_score(x_test, y_test))
    print(model.recall_score(x_test, y_test))
    print(model.precision_score(x_test, y_test))

def metrics(y_test , y_predict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_predict)):
        if y_predict[i] == 1:
            if y_test[i] == 1:
                TP += 1
            elif y_test[i] == 0:
                FP += 1
        if y_predict[i] == 0:
            if y_test[i] == 0:
                TN += 1
            elif y_test[i] == 1:
                FN += 1

    print('TP:{} , TN:{} , FP:{} , FN:{}'.format(TP , TN , FP , FN))
    print('recall:{} , precision:{}'.format(TP / (TP + FN) , TP / (TP + FP)))



if __name__ == '__main__':
    train_with_astnn()
    train_with_merged_feature()
    train_with_metrics()
