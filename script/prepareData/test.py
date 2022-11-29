from gensim.models.word2vec import Word2Vec
from prepareData.utils import get_blocks_v1 as func
import numpy as np
from prepareData.model import BatchProgramClassifier


class Parser:
    def parse(self , func):
        import javalang
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree

    def generate_block_seqs(self, node):
        word2vec = Word2Vec.load('vocab/node_w2v_' + str(128)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]
        print('----' + 'max_token:' +str(max_token) + '----')
        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        tree = trans2seq(node)
        print(tree)
        # print(len(tree))
        return tree

    def read(self , filePath):
        with open(filePath , 'r') as f:
            return f.read()

    def run(self , filePath):
        code = self.read(filePath)
        node = self.parse(code)
        blocks = self.generate_block_seqs(node)
        return blocks


# construct dictionary and train word embedding
    def dictionary_and_embedding(self, ast , size):
        from prepareData.utils import get_sequence as func
        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence
        corpus = [trans_to_sequences(ast)]
        print(corpus)
        # 更改配置项，这个代码写错了，也不是写错，需要把mincount配置项设置小一点
        w2v = Word2Vec(sentences=corpus, size=size, workers=16, sg=1, min_count=1,window=5, max_final_vocab=3000)
        # print(w2v.wv.vectors.shape)
        w2v.save('vocab/' + '/node_w2v_' + str(size))

def test1():
    parser = Parser()
    train = parser.parse(parser.read('testData\\Main.java'))
    for path , node in train:
        print(type(node))
        print(node)
        for child in node.children:
            print(type(child) , child)
        print(len(node.children))
    parser.dictionary_and_embedding(train, 128)
    # test = parser.parse(parser.read('Main2.java'))
    # parser.generate_block_seqs(test)
    #
    # train_data = parser.generate_block_seqs(parser.parse(parser.read('Main.java')))
    # train_data = [train_data]
    # test_data = parser.generate_block_seqs(parser.parse(parser.read('Main2.java')))
    # test_data = [test_data]
    #
    # word2vec = Word2Vec.load("vocab/node_w2v_128").wv
    # embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    # embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    #
    # HIDDEN_DIM = 100
    # ENCODE_DIM = 128
    # LABELS = 104
    # BATCH_SIZE = 1
    # USE_GPU = False
    # MAX_TOKENS = word2vec.syn0.shape[0]
    # EMBEDDING_DIM = word2vec.syn0.shape[1]
    #
    # model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
    #                                USE_GPU, embeddings)
    # if USE_GPU:
    #     model.cuda()
    # print('Start training...')
    # # training procedure
    # best_model = model
    # i = 0
    # while i < len(train_data):
    #     i += BATCH_SIZE
    #     # print('i:{} , len(train_data):{}'.format(i , len(train_data)))
    #     train_inputs = train_data
    #     model.zero_grad()
    #     print(train_inputs)
    #     model(train_inputs)
    # model = best_model
    # i = 0
    # while i < len(test_data):
    #     i += BATCH_SIZE
    #     test_inputs = test_data
    #     if USE_GPU:
    #         test_inputs = test_inputs
    #     model(test_inputs)

def test2():
    parser = Parser()
    train = parser.parse(parser.read('testData\\Main.java'))
    parser.dictionary_and_embedding(train, 128)
    test = parser.parse(parser.read('testData\\Main2.java'))
    parser.generate_block_seqs(test)

    train_data = parser.generate_block_seqs(parser.parse(parser.read('testData\\Main.java')))
    train_data = [train_data]
    test_data = parser.generate_block_seqs(parser.parse(parser.read('testData\\Main2.java')))
    train_data.append(test_data)
    val = parser.generate_block_seqs(parser.parse(parser.read('testData\\Main1.java')))
    train_data.append(val)

    word2vec = Word2Vec.load("vocab/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 104
    BATCH_SIZE = 3
    USE_GPU = False
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    print('Start training...')
    i = 0
    while i < len(train_data):
        print('{}-{} data start to train.'.format(i , i + BATCH_SIZE - 1))
        i += BATCH_SIZE
        # print('i:{} , len(train_data):{}'.format(i , len(train_data)))
        train_inputs = train_data
        model.zero_grad()
        res = model(train_inputs)
        print('out:{} + type:{} + shape:{}'.format(res , type(res) , res.shape))


if __name__ == '__main__':
    # test1()
    test2()




