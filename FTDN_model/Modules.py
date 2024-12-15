from time import time
from unicodedata import bidirectional
import torch
import torch.nn as nn
import utils
import torchvision

class Preprocess(nn.Module):
    """
    原始数据的预处理模块（一维卷积）
    :param n_features: 时间序列的特征数目
    :param kernel_size: 一维卷积的核大小
    """

    def __init__(self, n_features, kernel_size=7):
        super(Preprocess, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.padding(x).float()
        x = self.conv(x)
        x = self.relu(x)
        return x

class SpaceAttentionModule(nn.Module):
    """
    空间尺度上的图注意力模块
    :param n_features: 时间序列的特征数目
    :param time_step: 时间序列的时间步长
    :param dropout: dropout概率
    :param alpha: Leaky ReLU中使用的负斜率
    :param topk: 可学习图结构的连接节点数
    :param embed_dim: 特征尺度图注意力嵌入的维度
    :param use_bias: 是否带偏置
    :param use_prior_embed: 是否学习先验图结构
    :param device: 使用cpu还是gpu
    """ 
    def __init__(self, n_features, time_step, dropout, alpha, topk, embed_dim=None, use_bias=True, use_prior_embed=False, device='cpu'):
        super(SpaceAttentionModule, self).__init__()
        self.n_features = n_features
        self.time_step = time_step
        self.topk = topk
        self.embed_dim = embed_dim if embed_dim is not None else self.time_step
        self.embed_dim = self.embed_dim * 2
        self.lin_input_dim = 2 * time_step
        self.a_input_dim = self.embed_dim
        self.num_nodes = self.n_features
        self.use_bias = use_bias
        self.use_prior_embed = use_prior_embed
        self.device = device

        if not self.use_prior_embed:
            self.lin_transform = nn.Linear(self.lin_input_dim, self.embed_dim)
        else:
            self.lin_transform = nn.Linear(2 * self.lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty(self.a_input_dim, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(self.n_features, self.n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, embedding):
        a_input = self._make_attention_input(x, embedding, self.use_prior_embed)
        a_input = self.leakyrelu(self.lin_transform(a_input))
        e = torch.matmul(a_input, self.a).squeeze(3) 

        if self.use_bias:
            e += self.bias

        if self.use_prior_embed:
            learned_graph, adjacency_m = utils.LearnFeatureGraph(embedding, self.topk)#函数用于根据给定的先验嵌入学习序列特征的图结构
            zero_vec = -1e12 * torch.ones_like(e)
            attention = torch.where(adjacency_m.to(self.device) > 0, e, zero_vec) 
            attention = torch.softmax(attention, dim=2)
        else:
            attention = torch.softmax(e, dim=2)

        attention = self.dropout(attention).to(self.device)
        h_space = self.sigmoid(torch.matmul(attention, x))
        
        if self.use_prior_embed:
            return h_space, learned_graph
        else:
            return h_space, 0

    def _make_attention_input(self, v, embedding, use_prior_embed):
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  
        blocks_alternating = v.repeat(1, K, 1)  
        if not use_prior_embed:
            combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
            return combined.view(v.size(0), K, K, 2 * self.time_step)
        else:
            embedding_repeating = embedding.repeat_interleave(K, dim=0).unsqueeze(0)
            embedding_repeating = embedding_repeating.repeat(v.shape[0], 1, 1)
            embedding_alternating = embedding.repeat(K, 1).unsqueeze(0)
            embedding_alternating = embedding_alternating.repeat(v.shape[0], 1, 1)
            blocks_repeating = torch.cat((blocks_repeating, embedding_repeating), dim=2)
            blocks_alternating = torch.cat((blocks_alternating, embedding_alternating), dim=2)
            combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
            return combined.view(v.size(0), K, K, 4 * self.time_step)

class TemporalAttentionModule(nn.Module):
    """
    时间尺度上的图注意力模块
    :param n_features: 时间序列的特征数目
    :param time_step: 时间序列的时间步长
    :param dropout: dropout概率
    :param alpha: Leaky ReLU中使用的负斜率
    :param embed_dim: 特征尺度图注意力嵌入的维度
    :param use_bias: 是否带偏置
    """

    def __init__(self, n_features, time_step, dropout, alpha, embed_dim=None, use_bias=True, device='cpu'):
        super(TemporalAttentionModule, self).__init__()
        self.n_features = n_features
        self.time_step = time_step
        self.device = device
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.embed_dim = self.embed_dim * 2
        self.lin_input_dim = 2 * n_features
        self.a_input_dim = self.embed_dim
        self.num_nodes = self.time_step
        self.use_bias = use_bias
        
        self.lin_transform = nn.Linear(self.lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((self.a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(self.time_step, self.time_step))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        a_input = self._make_attention_input(x)              
        a_input = self.leakyrelu(self.lin_transform(a_input))        
        e = torch.matmul(a_input, self.a).squeeze(3)         

        if self.use_bias:
            e += self.bias  

        attention = torch.softmax(e, dim=2)
        attention = self.dropout(attention)

        h_time = self.sigmoid(torch.matmul(attention, x))    

        return h_time.permute(0, 2, 1)

    def _make_attention_input(self, v):
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  
        blocks_alternating = v.repeat(1, K, 1)  
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        
        return combined.view(v.size(0), K, K, 2 * self.n_features)
    
class GRUEncoder(nn.Module):
    """
    GRU编码器模块
    :param in_dim: 输入特征尺寸
    :param hid_dim: GRU隐藏层尺寸
    :param n_layers: GRU层数
    :param dropout: dropout概率
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRUEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.encoder = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, bidirectional=False, dropout=self.dropout)
        #self.encoder = nn.LSTM(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, h = self.encoder(x)
        out, h = out[-1, :, :], h[-1, :, :]  # GRU
        #out, h = out[-1, :, :], h[0][-1, :, :]  #LSTM
        return out, h

class GRUDecoder(nn.Module):
    """
    GRU解码器模块
    :param time_step: 重构序列的时间步长
    :param in_dim: 输入特征尺寸
    :param n_layers: GRU层数
    :param hid_dim: GRU隐藏层尺寸
    :param dropout: dropout概率
    """

    def __init__(self, time_step, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(GRUDecoder, self).__init__()
        self.time_step = time_step
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.decoder = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.time_step, dim=1).view(x.size(0), self.time_step, -1)

        decoder_out, _ = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out

class DeepfakeClassifier(nn.Module):

    def __init__(self, in_dim, hid_dim, dropout):
        super(DeepfakeClassifier, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim, int(hid_dim/2))
        self.linear4 = nn.Linear(int(hid_dim/2), 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.dropout(out)
        out = self.linear4(out)
        return self.sigmoid(out)

class FCEncoder(nn.Module):

    def __init__(self, feature_num, time_step, hid_dim, out_dim, dropout):
        super(FCEncoder, self).__init__()
        self.in_dim = feature_num * time_step * 1
        self.linear1 = nn.Linear(self.in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.relu(self.linear2(out))
        return out

class FCN(nn.Module):
    def __init__(self, feature_num, out_dim):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_num, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.bn1(out)
        out = self.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.relu(self.conv3(out))
        return out.squeeze(1)