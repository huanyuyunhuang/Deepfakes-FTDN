from ast import arg
import os
import utils
import torch
import numpy as np
import torch.nn as nn

from datetime import datetime
from args import get_parser         #args是什么（最重要问题）
from Train import Trainer
from Model import FTDN

if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    args = get_parser().parse_args()
    print(str(args.__dict__))
    #新data可以直接赋值28/29
    feature_num = 28
    if args.reconstruct_dim == None:
        args.reconstruct_dim = feature_num

    Log_dir, ModelPara_dir = utils.BuildOutputFolder(args.OutputRoot, id)#建立输出所需文件夹
    train_loader = utils.GetDataLoader(os.path.join(args.DataRoot, 'train_DF_c40'),
                                       args.train_batch_size, args.use_mean,
                                       args.use_median, args.use_mode,
                                       args.use_50, val=False, test=False)
    val_loader = utils.GetDataLoader(os.path.join(args.DataRoot, 'val_DF_c40'),
                                     args.val_batch_size, args.use_mean,
                                     args.use_median, args.use_mode,
                                     args.use_50, val=True, test=False)

    model = FTDN(
        feature_num = feature_num,
        time_step = args.time_step,
        topk = int(args.TopK * feature_num),#特征先验图结构保留的节点数目args.TopK * feature_num
        reconstruct_dim = args.reconstruct_dim,
        kernel_size = args.kernel_size,
        prior_embed_dim = args.prior_embed_dim,
        feat_gat_embed_dim = args.feat_gat_embed_dim,
        time_gat_embed_dim = args.time_gat_embed_dim,
        gru_n_layers = args.gru_n_layers,
        gru_hid_dim = args.gru_hid_dim,
        en_fc_n_layers = args.en_fc_n_layers,
        en_fc_hid_dim = args.en_fc_hid_dim,
        forecast_n_layers = args.fc_n_layers,
        forecast_hid_dim  = args.fc_hid_dim,
        recon_n_layers = args.recon_n_layers,
        recon_hid_dim = args.recon_hid_dim,
        dropout = args.dropout,
        alpha = args.alpha,
        use_prior_embed = args.use_prior_embed,
        use_cuda = args.use_cuda
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    forecast_criterion = nn.BCELoss(reduction='mean')
    recon_criterion = nn.MSELoss(reduction='mean')

    trainer = Trainer(
        model,
        optimizer,
        Feature_num = feature_num,
        Time_step = args.time_step,
        target_dims = None,
        n_epochs = args.epochs,
        forecast_criterion = forecast_criterion,
        recon_criterion = recon_criterion,
        log_dir = Log_dir,
        model_para_dir = ModelPara_dir
        )

    trainer.fit(train_loader, val_loader)