import os
class Config:
    # 方法体所在目录
    # "/root/GY/project/ReplaceSliceFunc"
    funcSourceDirPath = "D:\project\ReplaceSliceFunc"


    # funcAddSourceDirPath = "D:\\SliceAddFuncData"
    # 特征列
    attribute = ['package', 'version' , 'fileName', 'type', 'catogray', 'priority', 'rank' ,'start', 'end', 'label', 'code']
    # 'lineNum' , 'statementNum' , 'branchStatementNum' , 'callNum' , 'cycleComplexity' , 'depth'
    # 初始源码特征存放路径(源码仅经过切片，ast可编译处理)
    # 'data/program.pkl'
    programSourceInfoFilePath = 'solr/program.pkl'

    # programAddSourceInfoFilePath = './data/programAdd.pkl'

    # AST特征存放路径
    # 'data/ast.pkl'
    programASTFilePath = 'solr/ast.pkl'

    # programAddASTFilePath = './data/astAdd.pkl'
    # 度量属性
    metricsFilePath = os.path.join(funcSourceDirPath , 'metrics.xls')
    # 数据划分比例: train/test
    ratio = '8:2'
    # 词汇表长度
    vocabLength = 128