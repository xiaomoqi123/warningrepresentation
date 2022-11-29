import pandas as pd
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from prepareData.model import BatchProgramClassifier
from prepareData.test import Parser
from torch.autograd import Variable


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2]-1)
    return data, torch.LongTensor(labels)


if __name__ == '__main__':
    parser = Parser()
    train_data = parser.generate_block_seqs(parser.parse(parser.read('Main.java')))
    train_data = pd.DataFrame(train_data)
    test_data = parser.generate_block_seqs(parser.parse(parser.read('Main2.java')))
    test_data = pd.DataFrame(test_data)

    word2vec = Word2Vec.load("vocab/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 104
    BATCH_SIZE = 1
    USE_GPU = False
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    print('----model parameters----parameters:{},type(parameters):{}'.format(parameters , type(parameters)))
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    start_time = time.time()
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    while i < len(train_data):
        batch = get_batch(train_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        train_inputs, train_labels = batch
        if USE_GPU:
            train_inputs, train_labels = train_inputs, train_labels.cuda()

        model.zero_grad()
        model.batch_size = len(train_labels)
        model.hidden = model.init_hidden()
        output = model(train_inputs)

        loss = loss_function(output, Variable(train_labels))
        loss.backward()
        optimizer.step()

        # calc training acc
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == train_labels).sum()
        total += len(train_labels)
        total_loss += loss.item() * len(train_inputs)

    train_loss_.append(total_loss / total)
    train_acc_.append(total_acc.item() / total)
    # validation epoch
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
