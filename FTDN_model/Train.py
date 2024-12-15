import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    FTDN模型的训练类
    :param model: 训练的模型
    :param optimizer: 梯度下降使用的优化器
    :param Feature_num: 序列的特征数目
    :param Time_step: 序列的时间步长度
    :param target_dims: GRU输出的特征向量维度
    :param start_epoch: 起始的代数
    :param n_epochs: 迭代次数
    :param batch_size: 样本批次大小
    :param lr: 优化器学习率
    :param forecast_criterion: DeepFake分类损失函数
    :param recon_criterion: 序列重建损失函数
    :param boolean use_cuda: 是否使用GPU
    :param log_dir: 日志存储路径
    :param model_para_dir: 模型参数存储路径
    :param print_every: 每多少次迭代打印损失
    :param log_tensorboard: 是否在tensorboard上记录损失
    :param args_summary: 参数总结
    """

    def __init__(
        self,
        model,
        optimizer,
        Feature_num,
        Time_step,
        target_dims = None,
        start_epoch = 1,
        n_epochs = 200,
        batch_size = 64,
        lr = 0.001,
        forecast_criterion = nn.BCEWithLogitsLoss(),
        recon_criterion = nn.MSELoss(),
        use_cuda = False,
        log_dir = r'./result/LogFile',
        model_para_dir = r'./result/ModelPara',
        print_every = 1,
        log_tensorboard = True,
        args_summary = "",
    ):
        self.model = model
        self.optimizer = optimizer
        self.Feature_num = Feature_num
        self.Time_step = Time_step
        self.target_dims = target_dims
        self.start_epoch = start_epoch
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.log_dir = log_dir
        self.model_para_dir = model_para_dir 
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.loss_value = {
            "train_total": [],
            "train_classify": [],
            "train_reconstruct": [],
            "val_total": [],
            "val_classify": [],
            "val_reconstruct": [],
        }

        self.accuracy = {
            "train_acc": [],
            "val_acc": [],
        }

        self.epoch_times = []
        self.valbest = 0
        
        if self.device == 'cuda':
            # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
            # self.model = DataParallel(self.model.cuda(), device_ids=[0,1,2])
            self.model.cuda()

        if self.log_tensorboard == True:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)
    
    def train(self, train_loader):
        """
        模型训练函数，将训练损失存储在self.loss_value中
        :param train_loader: 训练数据
        """
        epoch_start_time = time.time()
        self.model.train()
        classify_loss = 0
        reconstruct_loss = 0
        total_loss = 0

        correct = torch.zeros(1).squeeze().to(self.device)
        total = torch.zeros(1).squeeze().to(self.device)

        for i, (data, label, video_log, idx) in enumerate(train_loader):
            data, label = data.to(self.device), label.to(self.device).float()
            self.optimizer.zero_grad()
            preds, recons = self.model(data)#爆显存
            c_loss = self.forecast_criterion(preds, label) 
            r_loss = self.recon_criterion(recons, data.float())
            loss = c_loss + r_loss
            loss.backward()
            self.optimizer.step()
                
            classify_loss += c_loss.item()
            reconstruct_loss += r_loss.item()
            total_loss += loss.item()

            prediction = torch.zeros_like(preds)
            for x in range(preds.shape[0]):
                if preds[x] > 0.5:
                    prediction[x] = 1
                else:
                    prediction[x] = 0
            correct += (prediction == label).sum().float()
            total += len(label)

            # if i % 10 == 0:
            print('Step:', i, 'c_loss:', c_loss.item(), 'r_loss', r_loss.item(),'total:', loss.item(),'acc:', correct/total)

        self.loss_value["train_classify"].append(classify_loss)
        self.loss_value["train_reconstruct"].append(reconstruct_loss)
        self.loss_value["train_total"].append(total_loss)
        self.accuracy["train_acc"].append(correct/total)

        epoch_time = time.time() - epoch_start_time

        return epoch_time

    def validation(self, val_loader):
        """
        模型训练函数，将验证损失存储在self.loss_value中
        :param val_loader: 验证数据
        """

        self.model.eval()
        classify_loss = 0
        reconstruct_loss = 0
        total_loss = 0

        correct = torch.zeros(1).squeeze().to(self.device)
        total = torch.zeros(1).squeeze().to(self.device)

        for i, (data, label, video_log, idx) in enumerate(val_loader):
            data, label = data.to(self.device), label.to(self.device).float()

            preds, recons = self.model(data)

            c_loss = self.forecast_criterion(preds, label) 
            r_loss = self.recon_criterion(recons, data.float())
            loss = c_loss + r_loss
                
            classify_loss += c_loss.item()
            reconstruct_loss += r_loss.item()
            total_loss += loss.item()

            prediction = torch.zeros_like(preds)
            for x in range(preds.shape[0]):
                if preds[x] > 0.5:
                    prediction[x] = 1
                else:
                    prediction[x] = 0
            correct += (prediction == label).sum().float()
            total += len(label)
            acc = (correct/total)

            if i % 1 == 0:
                print('Step:', i, 'c_loss:', c_loss.item(), 'r_loss', r_loss.item(),'total:', loss.item(),'acc:', acc)

        self.loss_value["val_classify"].append(classify_loss)
        self.loss_value["val_reconstruct"].append(reconstruct_loss)
        self.loss_value["val_total"].append(total_loss)
        self.accuracy["val_acc"].append(acc)

        if acc > self.valbest:
            self.valbest = acc
        print('val_best:', self.valbest)

    def save_modelpara(self, model, root, optimizer, epoch_num):
        """
        函数用于保存模型参数
        :param model: 保存的模型对象
        :param path: 存储根路径
        :param optimizer: 模型对象的优化器参数
        :param epoch_num: 模型对象训练迭代的次数
        """

        state = {'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_num}
        save_path = os.path.join(root, 'modelpara' + str(epoch_num) + '.pth')
        torch.save(state, save_path)

    def fit(self, train_loader, val_loader):
        """
        FTDN模型的拟合函数，用于对数据进行self.n_epochs次的训练和验证
        :param train_loader: 训练数据
        :param val_loader: 验证数据
        """

        start_time = time.time()
        for epoch in range(self.start_epoch, self.n_epochs + 1):
            
            print(f"Train model at {epoch} epochs..")
            epoch_time = self.train(train_loader)
            self.epoch_times.append(epoch_time)

            print(f"validate model at {epoch} epochs..")
            self.validation(val_loader)

            if self.log_tensorboard:
                self.write_message(epoch)

            if len(self.accuracy["val_acc"]) == 1:
                self.save_modelpara(self.model, self.model_para_dir, self.optimizer, epoch)
            elif self.accuracy["val_acc"][-1] >= self.valbest:
                self.save_modelpara(self.model, self.model_para_dir, self.optimizer, epoch)
            
            if epoch % self.print_every == 0:
                print('Printing Epoch: {:.0f}, Epoch_time: {:.0f}'.format(epoch, epoch_time))

                (print('Train_classify_loss: {:.4f}, Train_reconstruct_loss: {:.4f}, Train_total_loss: {:.4f}'.format
                (self.loss_value["train_classify"][-1], self.loss_value["train_reconstruct"][-1], self.loss_value["train_total"][-1])))

                (print('Val_classify_loss: {:.4f}, Val_reconstruct_loss: {:.4f}, Val_total_loss: {:.4f}'.format
                (self.loss_value["val_classify"][-1], self.loss_value["val_reconstruct"][-1], self.loss_value["val_total"][-1])))
        
        train_time = int(time.time() - start_time)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def write_message(self, epoch):
        for key, value in self.loss_value.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)
        for key, value in self.accuracy.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)

   