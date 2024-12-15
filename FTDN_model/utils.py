import os
from unittest import result
from matplotlib import use
import torch
import numpy as np

from Database import TSData, TSData_test
from torchvision import transforms
from torch.utils.data import DataLoader

def BuildOutputFolder(output_root, id):
    """
    函数用于建立程序输出所需的文件夹，返回日志和模型参数的存储路径
    :param output_root: 输出的根目录
    :param id: 当前的训练id
    """

    output_path = os.path.join(output_root, id)
    log_dir = os.path.join(output_path, 'LogFile')
    modelpara_dir = os.path.join(output_path, 'ModelPara')
    os.makedirs(output_path)
    os.makedirs(log_dir)
    os.makedirs(modelpara_dir)

    return log_dir, modelpara_dir

def GetDataLoader(Data_root, batch_size, use_mean, use_median, use_mode, use_50, val=False, test=False, num_worker=0):
    """
    函数用于建立训练和测试数据的DataLoader
    :param Data_root: 待用数据的根路径
    :param batch_size: 数据loader的批次大小
    :param boolean use_color: 是否使用颜色序列
    :param boolean use_trajectory: 是否使用区域轨迹序列
    :param boolean use_landmark: 是否使用landmark序列
    :param boolean val: 是否为验证集
    :param boolean test: 是否为测试集
    :param num_worker: 载入数据num_worker个数
    """

    if not test:
        if not val:
            dataset = TSData(root_dir=Data_root, Training=True, transform=transforms.ToTensor(),
                             use_mean=use_mean, use_median=use_median, use_mode=use_mode,
                             use_50=use_50)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        else:
            dataset = TSData(root_dir=Data_root, Training=False, transform=transforms.ToTensor(), use_mean=use_mean, use_median=use_median, use_mode=use_mode, use_50=use_50)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    else:
        dataset = TSData_test(root_dir=Data_root, Training=False, transform=transforms.ToTensor(), use_mean=use_mean, use_median=use_median, use_mode=use_mode, use_50=use_50)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    
    return dataloader


def LearnFeatureGraph(embedding, topk):
    """
    函数用于根据给定的先验嵌入学习序列特征的图结构
    :param embedding: 特征的先验嵌入
    :param topk: 选择前k个相关性最大节点建立图的边
    """
    weights = embedding.detach().clone()
    cos_ji_mat = torch.matmul(weights, weights.T)
    normed_mat = torch.matmul(weights.norm(dim = -1).view(-1, 1), weights.norm(dim = -1).view(1, -1))
    cos_ji_mat = cos_ji_mat / normed_mat

    topk_indices_ji = torch.topk(cos_ji_mat, topk, dim=-1)[1]

    adjacency_m = torch.zeros((embedding.shape[0], embedding.shape[0]))
    for i in range(topk_indices_ji.shape[0]):
        for j in range(topk_indices_ji.shape[1]):
            adjacency_m[i, topk_indices_ji[i, j]] = 1
            
    return topk_indices_ji.sort().values, adjacency_m

def SelectInputVector(v, graph):
    """
    函数用于根据学习到的特征图结构，返回相应节点的特征向量
    :param v: 特征向量
    :param graph: 学习到的特征图结构
    """

    result_arr = torch.zeros((v.shape[0], graph.shape[0] * graph.shape[1], v.shape[2]))
    new_x = torch.zeros((v.shape[0], graph.shape[1], v.shape[2]))
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            result_arr[:, i * graph.shape[1] + j, :] = v[:, graph[i, j], :]
            new_x[:, j, :] = v[:, graph[i, j], :]
    return result_arr

def AddZeros(att, graph):
    new_att = torch.zeros((att.shape[0], att.shape[1], att.shape[1]))
    for i in range(new_att.shape[1]):
        count = 0
        for j in range(new_att.shape[2]):
            if j in graph[i]:
                new_att[:, i, j] = att[:, i, count]
                count += 1
            else:
                new_att[:, i, j] = 0
    
    return new_att

