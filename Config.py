import torch
from Utils.Utils import LoadData, CrossEntropy2d, Receiver, PerturbationSpectrum
from torch import optim
from pathlib import Path
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
from typing import Tuple, Dict


class AffinityLoss(nn.Module):
    """
    Description: 用于语义分割任务的自定义损失函数类,核心目标是让模型不仅学习每个像素的类别(即"一元"信息),还学习像素之间的两两关系(亲和性)-->即两个像素是否属于同一语义类别.
                 传统分割损失(如交叉熵)只关注单个像素的预测是否正确,AffinityLoss引入了像素对之间的关系建模:如果两个像素在真实标签中属于同一类,则它们的"亲和值"应为1;否则为0.
    """

    def __init__(self, num_class, prior_map_size, ignore_label, reduction='mean', lambda_b=1.0, lambda_g=1.0, align_corners=False):
        """
        function      : Initialize
        num_class     : 类别总数
        prior_map_size: 亲和矩阵的空间尺寸(通常小于原图,[h/2,w/2]),用于降低计算复杂度
        ignore_label  : 标签中忽略的类别ID,不参与损失计算
        reduction     : 损失归约方式('mean'/'sum'/'none')
        lambda_b      : 一元损失权重
        lambda_g      : 全局指标损失总权重
        align_corners : 插值时是否对齐角落
        """
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.prior_map_size = prior_map_size
        self.reduction = reduction
        self.lambda_b = lambda_b
        self.lambda_g = lambda_g
        self.align_corners = align_corners

    def forward(self, prior_map, label):
        """
        function : Forward
        prior_map: (batch,height'*width',height'*width')
        label    : (batch,1,height,width)
        return   : loss value
        """
        # 理想亲和矩阵==>(batch,height'*width',height'*width')
        ideal_affinity_matrix = self.construct_ideal_affinity_matrix(label, self.prior_map_size, self.ignore_label)
        # 计算一元损失
        unary_loss = self.binary_cross_entropy(prior_map, ideal_affinity_matrix)
        # 构造对角掩码(排除自相关)
        diagonal_matrix = (1 - torch.eye(ideal_affinity_matrix.shape[1])).to(ideal_affinity_matrix.get_device())
        target = diagonal_matrix * ideal_affinity_matrix  # 对角线置零
        # 计算召回率损失
        recall_part = torch.sum(prior_map * target, dim=2)  # noqa
        denominator = torch.sum(target, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = self.binary_cross_entropy(recall_part, recall_label, reduction=self.reduction)
        # 计算特异性损失
        spec_part = torch.sum((1 - prior_map) * (1 - ideal_affinity_matrix), dim=2)
        denominator = torch.sum(1 - ideal_affinity_matrix, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = self.binary_cross_entropy(spec_part, spec_label, reduction=self.reduction)
        # 计算精确率损失
        precision_part = torch.sum(prior_map * target, dim=2)  # noqa
        denominator = torch.sum(prior_map, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = self.binary_cross_entropy(precision_part, precision_label, reduction=self.reduction)
        # 总损失
        global_loss = recall_loss + spec_loss + precision_loss
        return self.lambda_b * unary_loss + self.lambda_g * global_loss

    @staticmethod
    def binary_cross_entropy(predict_label, true_label, reduction='mean'):
        """
        function     : 计算二元交叉熵损失
        predict_label: 预测标签(B,H*W,H*W)
        true_label   : 真实标签(B,H*W,H*W)
        reduction    : ['none','mean','sum']
        return       : loss value
        """
        loss = func.binary_cross_entropy(predict_label, true_label.float(), weight=None, reduction='none')
        reduction_enum = func._Reduction.get_enum(reduction)  # noqa
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()

    def construct_ideal_affinity_matrix(self, label, label_size, ignore_label):
        """
        function  : 把语义分割标签label(每个像素一个类别),转换成一个"理想亲和矩阵"(ideal affinity matrix)
        label     : 真实标签==>[batch,1,height,width]
        label_size: 亲和图大小==>[height',width']
        return    : 理想亲和矩阵matrix==>[batch,height'*width',height'*width']

        Note      : matrix[b,i,j]=1 ==>位置i与j属于同一语义类别
                    matrix[b,i,j]=0 ==>位置i与j属于不同语义类别
        """
        label = torch.unsqueeze(label, dim=1)
        # 采样
        sample_label = func.interpolate(label.float(), size=label_size, mode="nearest")
        sample_label = torch.squeeze(sample_label, dim=1).long()
        sample_label[sample_label == ignore_label] = self.num_class
        # To one-hot
        one_hot_label = func.one_hot(sample_label, self.num_class + 1)
        one_hot_label = one_hot_label.view(one_hot_label.size(0), -1, self.num_class + 1).float()
        ideal_affinity_matrix = torch.bmm(one_hot_label, one_hot_label.permute(0, 2, 1))
        return ideal_affinity_matrix


class RCCALossFunction(nn.Module):
    def __init__(self, num_class, down_sample_size, ignore_label):
        """
        function        : Initialize
        num_class       : 类别个数
        down_sample_size: 先验图的高、宽值
        """
        super(RCCALossFunction, self).__init__()
        self.main_loss = CrossEntropy2d(ignore_label=ignore_label)
        self.affinity_loss = AffinityLoss(num_class=num_class, prior_map_size=down_sample_size, ignore_label=255)

    def forward(self, predict, prior_map, target):
        """
        function : Forward
        predict  : 预测概率:[batch,num_class,height,width]
        prior_map: 先验图:[batch,height'*width',height'*width']
        target   : 真实标签:[batch,num_class,height,width]
        return   : 损失值
        """
        loss = self.main_loss(predict, target) + self.affinity_loss(prior_map, target)
        return loss


def compute_loss(network_output: torch.Tensor, train_samples_gt_onehot: torch.Tensor, train_label_mask: torch.Tensor):
    real_labels = train_samples_gt_onehot
    we = -torch.mul(real_labels, torch.log(network_output))
    we = torch.mul(we, train_label_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy


def load_configuration(data_index: int, model: str, seed: int) -> Tuple[dict, Dict]:
    """🐳
    function    : 加载模型的训练设置
    data_index  : 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SA)
    model       : 模型名称
    seed        : 随机种子
    mode        : ["train", "test"]
    return      :
    """
    data_name = ["PaviaU", "Indian_pines_corrected", "Salinas_corrected"][data_index]
    loader = LoadData(data_name=data_name, data_path=fr"{Path(__file__).resolve().parent}/DataSet/{data_name}", is_print=False)
    num_class, channel, height, width = loader.number, loader.channel, loader.height, loader.width
    image_data, train_label, _ = loader.split_data(seeds=seed, number=100, except_label=except_label)
    # 模型特有参数
    prior_map_size = [int((((height - 12) / 2) - 12) / 2), int((((width - 12) / 2) - 12) / 2)]  # RCPNet所需的先验图大小
    data_id = {"PaviaU": 1, "Indian_pines": 3, "Salinas": 2, "Indian_pines_corrected": 3, "Salinas_corrected": 2, "Houston13": 4}[data_name]  # BANet所需ID
    dimension = {"PaviaU": 148 * 80, "Indian_pines_corrected": 31 * 31, "Salinas_corrected": 123 * 49}[data_name]
    # 模型初始化参数
    model_param = {"S3ANet": dict(num_features=channel, num_classes=num_class),
                   "S2Net": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width, filter_mode='adaptive'),
                   "RCPNet": dict(in_channel=channel, height=height, width=width, num_class=num_class),
                   "SACNet": dict(in_channel=channel, num_class=num_class),
                   "BANet": dict(num_features=channel, num_classes=num_class, dataID=data_id, attack=2, a=[4, 4, 4], k=[[3, 5, 7], [5, 7, 9], [3, 5, 7]], s=[3, 3, 2]),
                   "WFSS": dict(num_features=channel, num_classes=num_class, dim=dimension),
                   "Baseline": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width, ablation="Baseline", filter_mode='adaptive'),
                   "Baseline+SAC": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width, ablation="Baseline+SAC", filter_mode='adaptive'),
                   "Baseline+FDGP": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width, ablation="Baseline+FDGP", filter_mode='adaptive'),
                   "Baseline+SGS": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width, ablation="Baseline+SGS", filter_mode='adaptive'),
                   "Baseline+FDGP+SAC": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width, ablation="Baseline+FDGP+SAC", filter_mode='adaptive'),
                   "Baseline+SGS+FDGP": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width, ablation="Baseline+SGS+FDGP", filter_mode='adaptive'),
                   "Baseline+SGS+SAC": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width, ablation="Baseline+SGS+SAC", filter_mode='adaptive'),
                   "Baseline+SGS+FDGP+SAC": dict(in_channel=channel, embed=64, num_class=num_class, height=height, width=width,
                                                 ablation="Baseline+SGS+FDGP+SAC", filter_mode='adaptive'),
                   "CeGCN": dict(height=height, width=width, in_channel=channel, num_class=num_class, image_data=image_data, train_label=train_label),
                   "WFCG": dict(height=height, width=width, in_channel=channel, num_class=num_class, image_data=image_data, train_label=train_label),
                   "SsFCN": dict(in_channel=channel, num_class=num_class),
                   "PSFormer": dict(channel=channel, num_class=num_class, data_name=data_name, stage_num=4, encoder_num=3), }
    # 所有模型的训练设置(优化器、学习率、是否调整学习率、权重衰减、损失函数、训练轮数)
    train_settings = {"S2Net": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "Baseline": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                   "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "Baseline+SGS": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                       "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "Baseline+SAC": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                       "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "Baseline+FDGP": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                        "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "Baseline+SGS+FDGP": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                            "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "Baseline+SGS+SAC": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                           "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "Baseline+FDGP+SAC": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                            "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "Baseline+SGS+FDGP+SAC": {"optimizer": optim.AdamW, "lr": 5e-4, "adjust_learning_rate": False, "weight_decay": 1e-5,
                                                "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 200, "model_type": "General", 'num_class': num_class},
                      "RCPNet": {"optimizer": optim.Adam, "lr": 5e-4, "adjust_learning_rate": True, "weight_decay": 1e-5, "model_type": "RCPNet", 'num_class': num_class,
                                 "loss_func": RCCALossFunction(num_class=num_class, down_sample_size=prior_map_size, ignore_label=except_label), "epoch": 500},
                      "SACNet": {"optimizer": optim.Adam, "lr": 5e-4, "adjust_learning_rate": True, "weight_decay": 1e-5,
                                 "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 1000, "model_type": "General", 'num_class': num_class},
                      "BANet": {"optimizer": optim.Adam, "lr": 5e-4, "adjust_learning_rate": True, "weight_decay": 1e-5,
                                "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 1000, "model_type": "BANet", 'num_class': num_class},
                      "S3ANet": {"optimizer": optim.Adam, "lr": 5e-4, "adjust_learning_rate": True, "weight_decay": 1e-5,
                                 "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 1000, "model_type": "General", 'num_class': num_class},
                      "WFSS": {"optimizer": optim.Adam, "lr": 5e-4, "adjust_learning_rate": True, "weight_decay": 1e-5,
                               "loss_func": CrossEntropy2d(ignore_label=except_label), "epoch": 500, "model_type": "General", 'num_class': num_class},
                      "CeGCN": {"optimizer": optim.Adam, "lr": 5e-4, "weight_decay": 1e-5, "adjust_learning_rate": False, "loss_func": CrossEntropy2d(ignore_label=except_label),
                                "epoch": 600, "model_type": "General", 'num_class': num_class},
                      "WFCG": {"optimizer": optim.Adam, "lr": 5e-4, "weight_decay": 1e-5, "adjust_learning_rate": False, "loss_func": compute_loss,
                               "epoch": 300, "model_type": "Graph", 'num_class': num_class},
                      "SsFCN": {"optimizer": optim.Adam, "lr": 5e-4, "weight_decay": 1e-5, "adjust_learning_rate": True, "loss_func": CrossEntropy2d(ignore_label=except_label),
                                "epoch": 1000, "model_type": "General", 'num_class': num_class},
                      "PSFormer": {"optimizer": optim.Adam, "lr": 5e-4, "weight_decay": 1e-5, "adjust_learning_rate": False, "loss_func": CrossEntropy2d(ignore_label=except_label),
                                   "epoch": 600, "model_type": "General", 'num_class': num_class}, }
    return model_param[model], train_settings[model]


def generate_adv_example(data_name: str, model_name: str, epsilon: float, ignore_label: int, train_setting: dict, method="PGD") -> np.ndarray:
    """🐳
    function     : 基于给定的模型和数据集,生成对应扰动幅度条件下的对抗样本.
    data_name    : 数据集名称
    model_name   : 模型名称
    epsilon      : 扰动幅度
    ignore_label : 忽略的标签
    train_setting: 训练设置
    method       : 攻击方法(目前仅支持FGSM、PGD)
    return       : 对抗样本,形状为(channel,height,width),数据类型为numpy数组
    """
    if method not in ['PGD', "FGSM", "CW"]:
        raise ValueError("攻击方式写错辣!!!!!!")
    script_path = Path(__file__).resolve().parent
    # 加载图像、标签数据
    file_name = file_names[data_name]
    image = sio.loadmat(fr"{script_path}/DataSet/{data_name}/{data_name}.mat")[image_keys[data_name]]
    label = sio.loadmat(fr"{script_path}/DataSet/{data_name}/{file_name}.mat")[label_keys[data_name]]
    # 转为tensor数据
    image = torch.from_numpy(LoadData.norm_data(image).transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)
    label = torch.from_numpy(label[None, :, :]).to(device)
    # 加载模型
    model_path = fr"{script_path}/Result/Weight/{model_name}/{data_name}/1.pt"
    model = torch.load(model_path, weights_only=False).to(device)
    # 设置模型接收器
    receiver = Receiver(model_name=model_name, save_path=rf"{script_path}/Result", data_name=data_name)
    receiver.reset(model=model, except_label=ignore_label, settings=train_setting)
    # 生成对抗样本
    print(f"攻击模型:{model_name}")
    # 对抗样本形状==>{"method-epsilon":numpy.array(chanel,height,width),...,"method-epsilon":numpy.array(chanel,height,width)}
    if method == "PGD":
        adv_image = receiver.pgd_attack(image=image, true_label=label, root_dir="", epsilons=[epsilon])
    elif method == "FGSM":
        adv_image = receiver.target_fgsm_with_norm(image=image, label=label, root_dir="", epsilons=[epsilon])
    else:
        adv_image = receiver.cw_attack(image=image, true_label=label, root_dir="", epsilons=[epsilon])
    return adv_image[fr"{method}-{epsilon}"]


def generate_perturbation(data_index: int, model_names: list[str], method: str, epsilon: float) -> dict[str, np.ndarray]:
    """🐳
    function   : 根据模型的权重获取对抗扰动数据
    data_index : 数据集索引:0⇒(PU)、1⇒(IP)、2⇒(SV)
    model_names: 模型名称:[model1,...,modelN]
    method     : 攻击方式
    epsilon    : 扰动幅度
    return     : {'model': np.array[channel,height,width], ...,}
    """
    script_path = Path(__file__).resolve().parent
    data_name = data_names[data_index]
    result = {}
    for name in model_names:
        _, train_setting = load_configuration(data_index=data_index, model=name, seed=5211314)
        # 加载图像、标签数据
        image = sio.loadmat(fr"{script_path}/DataSet/{data_name}/{data_name}.mat")[image_keys[data_name]]
        label = sio.loadmat(fr"{script_path}/DataSet/{data_name}/{file_names[data_name]}.mat")[label_keys[data_name]]
        model_path = fr"{script_path}/Result/Weight/{name}/{data_name}/1.pt"  # 权重地址
        receiver = Receiver(model_name=name, save_path=rf"{script_path}/Result", data_name=data_name)  # 模型接收器
        analysis = PerturbationSpectrum(receiver=receiver, clean_data=image, true_label=label, model_name=name, model_path=model_path,
                                        epsilon=epsilon, train_setting=train_setting, data_name=data_name)
        perturbation = analysis.gen_adv_content(method=method, mode='noise')  # (channel,height,width)
        torch.cuda.empty_cache()
        result[f'{name}'] = perturbation
    return result


# 高光谱数据集图像数据key
image_keys = {"PaviaU": "paviaU", "Indian_pines_corrected": "indian_pines_corrected", "Salinas_corrected": "salinas_corrected"}
# 高光谱数据集标签数据key
label_keys = {"PaviaU": "paviaU_gt", "Indian_pines_corrected": "indian_pines_gt", "Salinas_corrected": "salinas_gt"}
# 高光谱数据集标签数据文件名
file_names = {"PaviaU": "PaviaU_gt", "Indian_pines_corrected": "Indian_pines_gt", "Salinas_corrected": "Salinas_gt"}
# 高光谱数据集名称
data_names = ["PaviaU", "Indian_pines_corrected", "Salinas_corrected"]
# 消融实验模块组合
combined_modules = ["Baseline", "Baseline+FDGP+SAC", "Baseline+SGS+FDGP", "Baseline+SGS+SAC", "Baseline+SGS+FDGP+SAC"]  # noqa
# noqa ["Baseline", "Baseline+SGS", "Baseline+SAC", "Baseline+FDGP", "Baseline+FDGP+SAC", "Baseline+SGS+FDGP", "Baseline+SGS+SAC", "Baseline+SGS+FDGP+SAC"]
# 训练样本个数==>1.PU、2.IP、3.SA
train_numbers = [100, [20, 80, 80, 80, 80, 80, 15, 80, 10, 80, 80, 80, 80, 80, 50, 50], 100]
# 背景元素设置的标签
except_label = 255
# 类别个数
num_classes = {"PaviaU": 9, "Indian_pines_corrected": 16, "Salinas_corrected": 16}
# 随机种子
seeds = [202501, 202502, 202503, 202504, 202505, 202506, 202507, 202508, 202509, 202510]
# 不同攻击方法使用的扰动幅度
fgsm_epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
pgd_epsilons = [0.05]
cw_epsilons = [0.05]
# 是否使用对抗训练
against_train = {"S2Net": False, "S3ANet": False, "SACNet": False, "RCPNet": False, "WFSS": False, "Baseline": False, "Baseline+SGS": False,
                 "Baseline+SAC": False, "Baseline+FDGP": False, "Baseline+FDGP+SAC": False, "Baseline+SGS+FDGP": False, "Baseline+SGS+FDGP+SAC": False,
                 "Baseline+SGS+SAC": False, "CeGCN": True, "WFCG": True, "PSFormer": False, "SsFCN": True}
# 数据形状
data_shapes = {"PaviaU": (610, 340), "Indian_pines_corrected": (145, 145), "Salinas_corrected": (512, 217)}
# 颜色修饰
blue = '\033[34m'
green = '\033[32m'
red = '\033[31m'
reset = '\033[0m'
# 设备地址
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
