from prepareData.config import Config
from prepareData.service.preProcess import preProcess
import pandas as pd
import os
class Pipeline:
    def __init__(self):
        self.ratio = Config.ratio
        self.sources = preProcess().readASTFeatures()
        self.root = './'
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = Config.vocabLength

    def splitData(self):
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)
        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = self.root + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = self.root + 'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path + 'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = self.root + 'test/'
        check_or_create(test_path)
        self.test_file_path = test_path + 'test_.pkl'
        test.to_pickle(self.test_file_path)


    def dictionary_and_embedding(self, input_file):
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        from prepareData.utils import get_sequence

        def trans_to_sequences(ast):
            sequence = []
            # 获得ast的token序列,ast即FileAST根节点
            get_sequence(ast, sequence)
            return sequence
        # for index , value in trees['code'].items():
        #     print(' '.join(trans_to_sequences(value)))
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=self.size, workers=16, sg=1, min_count=3)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(self.size))


    def generate_block_seqs(self,data_path,part):
        from prepareData.utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec
        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        # 语料库的词汇个数 ， 猜测存疑
        # 词向量的大矩阵，第i行表示vocab中下标为i的词
        max_token = word2vec.syn0.shape[0]
        print(max_token)

        # blocknode
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
        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(self.root+part+'/blocks.pkl')

    def run(self):
        print('start split data...')
        self.splitData()
        print('train word embedding...')
        self.dictionary_and_embedding(None)
        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')

if __name__ == '__main__':
    Pipeline().run()
    print(pd.read_pickle('dev/blocks.pkl'))
