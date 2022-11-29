import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    # node:语句数组,batch_index:当前处理语句对应编号
    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None

        # 语句当前层编码
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        # index:起始为0，d为1的序号排序数组，主要用处是为了后面构建当层编码表示做一个映射
        # children_index:层次结构表示当前语句所属的相对位置，需要跟batch_index 一起联合使用
        index, children_index = [], []
        # current:当前层节点在嵌入层中的编号
        # children:层次结构表示所有子节点
        current_node, children = [], []
        for i in range(size):
            if node[i][0] is not -1:
                # 将自己加入index中
                index.append(i)
                current_node.append(node[i][0])
                # 直接子节点
                temp = node[i][1:]
                # 直接子节点的个数
                c_num = len(temp)
                # print('child_num:{}'.format(c_num))
                for j in range(c_num):
                    # 直接子节点存在
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        # 当前层的向量表示,采用相对位置数组index
        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        print('index:{} , children_index:{} , current_node:{} , children:{}'.format(index,
                                                                                    children_index , current_node , children) )
        print('batch_index:{}'.format(batch_index))
        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                # 这边代码逻辑好像有点问题，个人感觉应该是batch_children_index，但是看了前面的index，貌似没啥问题。用的是相对编号。
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)

        # 当前批次编号数组
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))

        # 将当前批次的节点添加到nodelist中，因为是递归添加，所以最终结果需要用torch.max取最大值
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        # 该批次所有语句的个数
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BatchProgramClassifier(nn.Module):
    # def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BatchProgramClassifier, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        #class "BatchTreeEncoder"
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        # 数组:tree中block数量
        lens = [len(item) for item in x]
        max_len = max(lens)

        # 一维数组,每一个元素代表语句所包含的节点index,即block
        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            # 标准化每个ast的长度为最大语句长度，便于使用gru做神经网络处理，一大亮点。针对语句个数不一致
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            # print(end - start)
            start = end
        encodes = torch.cat(seq)
        # print(encodes.shape)
        encodes = encodes.view(self.batch_size, max_len, -1)


        print('encodes:{} , shape(encodes):{}'.format(encodes , encodes.shape))

        # gru 重新设置了隐藏层(输出)的大小 ,初始化为100
        gru_out, hidden = self.bigru(encodes, self.hidden)
        print(hidden.shape)

        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        print('gru_out:{} , gru_out.size():{} , gru_out.shape():{}'.format(gru_out , gru_out.size(2) , gru_out.shape))
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]

        print('gru_out:{} , type(gru_out):{} , len(gru_out):{}'.format(gru_out , type(gru_out) , gru_out.shape))

        # linear
        # y = self.hidden2label(gru_out)
        return gru_out

