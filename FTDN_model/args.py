import argparse

from numpy import False_


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # -- 数据参数 ---
    parser.add_argument("--DataRoot", type=str, default=r"../../Data/USEDATA")
    parser.add_argument("--OutputRoot", type=str, default=r"../result")
    parser.add_argument("--use_mean", type=str2bool, default=True)
    parser.add_argument("--use_median", type=str2bool, default=True)
    parser.add_argument("--use_mode", type=str2bool, default=False)
    parser.add_argument("--use_50", type=str2bool, default=True)
    parser.add_argument("--time_step", type=int, default=64)#时间步长
    parser.add_argument("--normalize", type=str2bool, default=False)
    parser.add_argument("--detrend", type=str2bool, default=False)

    # -- 模型参数 ---
    # 图注意力嵌入
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    # GRU层
    parser.add_argument("--gru_n_layers", type=int, default=5)
    parser.add_argument("--gru_hid_dim", type=int, default=256)
    # 全连接编码层
    parser.add_argument("--en_fc_n_layers", type=int, default=1)
    parser.add_argument("--en_fc_hid_dim", type=int, default=1024)
    # 预测模型部分
    parser.add_argument("--fc_n_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=1024)
    # 重建模型部分
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=256)
    # 其它
    parser.add_argument("--alpha", type=float, default=0.2)

    # --- 训练参数 ---
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=2800)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--use_prior_embed", type=str2bool, default=False)
    parser.add_argument("--TopK", type=float, default=0.7)#Topk: 特征先验图结构保留的节点数目比例
    parser.add_argument("--prior_embed_dim", type=int, default=64)
    parser.add_argument("--reconstruct_dim", type=int, default=None)
    parser.add_argument("--use_cuda", type=str2bool, default=False)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)

    # --- 其它 ---
    parser.add_argument("--comment", type=str, default="")

    return parser