 
import os
import numpy as np
from Modules import *
import torch
import utils
import matplotlib.pyplot as plt
import torch.nn as nn
import math

class FTDN(nn.Module):
    """
    FTDN的模型定义类
    :param feature_num: 时间序列的特征个数
    :param time_step: 时间序列的时间步长
    :param topk: 特征先验图结构保留的节点数目
    :param reconstruct_dim: 重构的特征数量
    :param kernel_size: 一维卷积核尺寸
    :param prior_embed_dim: 特征先验嵌入的维度
    :param feat_gat_embed_dim: 特征尺度图注意力嵌入的维度
    :param time_gat_embed_dim: 时间尺度图注意力嵌入的维度
    :param gru_n_layers: GRU的层数
    :param gru_hid_dim: GRU隐藏层的维度
    :param en_fc_n_layers: 全连接编码的层数
    :param en_fc_hid_dim: 全连接编码隐藏层的维度
    :param forecast_n_layers: 基于全连接的预测模型的层数
    :param forecast_hid_dim: 基于全连接的预测模型的隐藏层维度
    :param recon_n_layers: 基于GRU的重构模型的层数
    :param recon_hid_dim: 基于GRU的重构模型的隐藏层维度
    :param dropout: dropout的概率
    :param alpha: Leaky ReLU中使用的负斜率
    :param boolean use_prior_embed: 是否使用可学习的先验图结构
    :param boolean use_cuda: 是否使用GPU
    """
    def __init__(
        self, 
        feature_num,
        time_step,
        topk,
        reconstruct_dim = 1,  
        kernel_size = 7,
        prior_embed_dim = 64,
        feat_gat_embed_dim = None,
        time_gat_embed_dim = None,
        gru_n_layers = 3,
        gru_hid_dim = 256,
        en_fc_n_layers = 1,
        en_fc_hid_dim = 1024,
        forecast_n_layers = 3,
        forecast_hid_dim = 1024,
        recon_n_layers = 1,
        recon_hid_dim = 256,
        dropout = 0.2,
        alpha = 0.2,
        use_prior_embed = False,
        use_cuda = True
        ):

        super(FTDN, self).__init__()

        self.feature_num = feature_num
        self.time_step = time_step
        self.topk = topk
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.learned_graph = None
        self.prior_embedding = nn.Embedding(feature_num, prior_embed_dim)
        nn.init.kaiming_uniform_(self.prior_embedding.weight, a=math.sqrt(5))
        #原始数据的预处理模块（一维卷积）
        self.data_preprocessing = Preprocess(self.feature_num, kernel_size)
        #空间尺度上的图注意力模块
        self.space_gat = SpaceAttentionModule(self.feature_num, self.time_step, dropout,
                                              alpha, self.topk, feat_gat_embed_dim,
                                              use_prior_embed=use_prior_embed, device=self.device)
        #时间尺度上的图注意力模块
        self.time_gat = TemporalAttentionModule(self.feature_num, self.time_step, dropout,
                                                alpha, time_gat_embed_dim, device=self.device)
        #GRU编码器模块
        self.gruencoder = GRUEncoder(2*self.feature_num, gru_hid_dim, gru_n_layers, dropout)
        #FC编码器模块
        self.fcencoder = FCEncoder(self.feature_num, self.time_step,
                                   en_fc_hid_dim, gru_hid_dim, dropout)
        #GRU解码器模块
        self.grudecoder = GRUDecoder(self.time_step, gru_hid_dim, recon_hid_dim,
                                     reconstruct_dim, recon_n_layers, dropout)
        #Deepfake分类器模块
        self.classifier = DeepfakeClassifier(2*gru_hid_dim, forecast_hid_dim, dropout)
        self.fcn = FCN(1, out_dim=gru_hid_dim)
        
    
    def forward(self, x):
        #data = self.data_preprocessing(x)
        #fcn_data = self.fcn(x.float())
        prior_embedding = self.prior_embedding(torch.arange(self.feature_num).to(self.device))
        h_space, self.learned_graph = self.space_gat(x.float(), prior_embedding)
        h_time = self.time_gat(x.float())
        h_concate = torch.cat([x.float(), h_time], dim=1)
        _, T_representation = self.gruencoder(h_concate)
        #_, T_representation = self.gruencoder(x.float())
        reconstruct_data = self.grudecoder(T_representation)
        #T_representation = self.fcn(T_representation.unsqueeze(1))
        S_representation = self.fcencoder(h_space)
    
        representation = torch.cat([T_representation, S_representation], dim=1)
        predicate = self.classifier(representation)
        #predicate = self.classifier(T_representation)
        
        return predicate.squeeze(-1), reconstruct_data.permute(0, 2, 1)
    
