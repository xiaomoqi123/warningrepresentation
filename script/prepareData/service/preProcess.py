import os
import pandas as pd
from prepareData.config import Config
import javalang
import numpy as np

import javalang.tree

class preProcess:

    def toPickle(self):
        fileNames = self.readDir()
        # sourcemonitor
        # metrics = self.readMetrics()
        features = [fileName.split('#') for fileName in  fileNames]
        lables = [feature[len(feature) - 1].split('.')[0] for feature in features]
        fileTexts = [open(os.path.join(Config.funcSourceDirPath ,fileName)).read() for fileName in fileNames]
        for i in range(0 , len(features)):
            feature = features[i]
            feature[len(feature) - 1] = lables[i]
            feature.append(fileTexts[i])
            # fileName = fileNames[i]
            # metricsFeature = metrics[metrics['fileName'] == fileName]
            # feature.append(metricsFeature.iloc[0]['lineNum'])
            # feature.append(metricsFeature.iloc[0]['statementNum'])
            # feature.append(metricsFeature.iloc[0]['branchStatementNum'])
            # feature.append(metricsFeature.iloc[0]['callNum'])
            # feature.append(metricsFeature.iloc[0]['cycleComplexity'])
            # feature.append(metricsFeature.iloc[0]['depth'])
        dataFrame = pd.DataFrame(features)
        dataFrame.columns = Config.attribute
        dataFrame.to_pickle(Config.programSourceInfoFilePath)

    def parse(self , func):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree

    def generateFeatures(self):
        dataFrame = pd.read_pickle(Config.programSourceInfoFilePath)
        asts = []
        index = []
        for _, item in dataFrame.iterrows():
            try:
                tree = self.parse(item["code"])
                asts.append(tree)
            except Exception as ex:
                print("出现如下异常%s"%ex)
                index.append(_)
                continue
        # dataFrame['code'] = dataFrame['code'].apply(self.parse)
        dataFrame.drop(index=index , inplace=True)
        # 索引重排很重要
        dataFrame = dataFrame.reset_index(drop=True)
        dataFrame['code'] = pd.Series(asts)
        dataFrame.to_pickle(Config.programASTFilePath)

    def readASTFeatures(self):
        dataFrame = pd.read_pickle(Config.programASTFilePath)
        return dataFrame

    def readSourceCode(self):
        dataFrame = pd.read_pickle(Config.programSourceInfoFilePath)
        return dataFrame

    def readMetrics(self):
        dataFrame = pd.read_excel(Config.metricsFilePath)
        return dataFrame


    def readDir(self):
        res = []
        for root,dirs,files in os.walk(Config.funcSourceDirPath):
            for file in  files:
                if file.endswith('java'):
                    res.append(file)
        return res

    def run(self):
        self.toPickle()
        self.generateFeatures()

if __name__ == '__main__':
    preProcess().run()
    print(preProcess().readSourceCode())
    print(preProcess().readASTFeatures())

    # dataframe = preProcess().readASTFeatures()
    #
    # print(preProcess().readASTFeatures().loc[6716]['code'])

    # df = pd.DataFrame(np.arange(20).reshape(5, 4),
    #                   columns=['A', 'B', 'C', 'D'])
    # df.drop([2 , 4], inplace=True)
    # df['D'] = pd.DataFrame([4, 5 ,9])
    # print(df)


