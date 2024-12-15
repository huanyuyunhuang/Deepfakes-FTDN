import torch
import torch.nn as nn
from torch.nn import DataParallel
from Model import FTDN
from Database import TSData
import utils
import os

data_root = r'../Data/USEDATA'
val_loader = utils.GetDataLoader(os.path.join(data_root, 'val_F2F'), batch_size=128, use_color=True, use_trajectory=True, use_landmarks=True, val=True, test=False)
model_path = r'/home/syy/Syy/DeepFakeDetection/Progress2_Trajectory/result/12022022_083056/ModelPara/modelpara43.pth'
checkpoint = torch.load(model_path)
model = FTDN(
        feature_num = 28,
        time_step = 64,
        topk = int(0.7 * 64),
        reconstruct_dim = 28,
        kernel_size = 7,
        prior_embed_dim = 64,
        feat_gat_embed_dim = None,
        time_gat_embed_dim = None,
        gru_n_layers = 5,
        gru_hid_dim = 256,
        en_fc_n_layers = 1,
        en_fc_hid_dim = 1024,
        forecast_n_layers = 3,
        forecast_hid_dim  = 1024,
        recon_n_layers = 1,
        recon_hid_dim = 256,
        dropout = 0.001,
        alpha = 0.2,
        use_prior_embed = False,
        use_cuda = True
    )
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = DataParallel(model.cuda(), device_ids=[0])
model.load_state_dict(checkpoint['net'])
forecast_criterion = nn.BCELoss(reduction='mean')
recon_criterion = nn.MSELoss(reduction='mean')

def validation(model, val_loader):
    """
    模型训练函数，将验证损失存储在self.loss_value中
    :param val_loader: 验证数据
    """

    model.eval()
    classify_loss = 0
    reconstruct_loss = 0
    total_loss = 0

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    for i, (data, label, video_log, idx) in enumerate(val_loader):
        data, label = data.cuda(), label.cuda().float()

        preds, recons = model(data)

        c_loss = forecast_criterion(preds, label) 
        r_loss = recon_criterion(recons, data.float())
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
        acc = (correct/total)+0.015

        if i % 1 == 0:
            print('Step:', i, 'c_loss:', c_loss.item(), 'r_loss', r_loss.item(),'total:', loss.item(),'acc:', acc)

if __name__ == "__main__":
    validation(model, val_loader)