import datetime
import json
import math
import os
import os.path
import re
import subprocess
import sys
import time
from typing import Tuple, Callable, Union, Dict, List
from matplotlib.ticker import MultipleLocator
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as func
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from skimage import exposure
from skimage.segmentation import slic
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.autograd import Variable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from pathlib import Path

# 高光谱数据集图像数据key
image_keys = {"PaviaU": "paviaU", "Indian_pines_corrected": "indian_pines_corrected", "Salinas_corrected": "salinas_corrected"}
# 高光谱数据集标签数据key
label_keys = {"PaviaU": "paviaU_gt", "Indian_pines_corrected": "indian_pines_gt", "Salinas_corrected": "salinas_gt"}
# 高光谱数据集标签数据文件名
file_names = {"PaviaU": "PaviaU_gt", "Indian_pines_corrected": "Indian_pines_gt", "Salinas_corrected": "Salinas_gt"}


class ColorProgressBar:
    """💞
    Description: 自定义进度条
    """

    def __init__(self, length, description=None, epoches=None, bar_length=50, float_length=7, newline=True, style=("=", ">", "o")):
        """
        function    : 初始化
        length      : 单个epoch循环次数
        description : 关于进度条的描述
        epoches     : 多少个epoch:总进度
        bar_length  : 进度条长度
        float_length: 浮点数显示位数
        newline     : 进度条完成后是否换行
        style       : ("💘","🏹","💖")、("=", ">", "o")、("🌕", "🚀", "🌎")
        """
        # 循环总次数
        self.max_num = int(math.log10(length)) + 1  # 进度条的对齐
        self.length = length  # 进度条循环次数
        self.description = description
        self.epoches = epoches  # 总进程循环次数
        self.epoches_length = None if epoches is None else int(math.log10(epoches)) + 1  # 总进程的对齐
        self.bar_length = bar_length
        self.float_length = float_length
        self.newline = newline
        self.style = style

    @staticmethod
    def color_print(string: str, decorate: str, printf: bool = False):
        """🖤
        function : 彩色打印字符串string
        string   : 待显示的字符串
        decorate : 如何修饰字符串string
        printf   : 是否立即显示:printf=True==>立即显示,printf=False==>返回该彩色字符串
        """
        setting = {'f_g_k': 30, 'f_g_r': 31, 'f_g_g': 32, 'f_g_y': 33, 'f_g_b': 34, 'f_g_p': 35, 'f_g_c': 36,
                   'f_g_w': 37,
                   'b_g_k': 40, 'b_g_r': 41, 'b_g_g': 42, 'b_g_y': 43, 'b_g_b': 44, 'b_g_p': 45, 'b_g_c': 46,
                   'b_g_w': 47,
                   'close': 0, 'height_light': 1, 'half_light': 2, 'italic': 3, 'underline': 4, 'flicker': 5,
                   'anti_dis': 7, 'black_out': 8}
        color_string = f"\033[{setting[decorate]}m{string}\033[0m"
        if printf is True:
            print(color_string)
        else:
            return color_string

    def show_progress_bar(self, index, message, epoch=None) -> None:
        """🤍
        function: 显示进度条
        index   : 在当前训练epoch中的进度
        epoch   : 训练轮数,默认值为None(只训练一轮==>数据一次便可以训练完成),若全部数据需要分批次训练,index就是训练的进度,epoch就是
                  所有数据训练完成的轮数
        message : 添加你想要显示的参数
                   example:
                   message=[['name',value,type],...,['accuracy',accuracy_value,float],['loss',loss_value,float]]
                   type只支持int、float
        return  : None
        """
        # 描述+epoch/epoches
        str1 = '' if self.description is None else self.color_print(self.description, 'f_g_r') + ":"
        str2 = '' if epoch is None else ("epoch:" + self.color_print(f"{epoch:0{self.epoches_length}d}", 'f_g_b') + '/'
                                         + self.color_print(f"{self.epoches:0{self.epoches_length}d}", 'f_g_r')) + " "
        # 当前进度/总长度[
        num1 = self.color_print(f"{index:0{self.max_num}d}", 'f_g_g')
        num2 = self.color_print(self.length, 'f_g_r')
        string = "\r" + str1 + str2 + num1 + self.color_print("/", 'f_g_b') + num2 + self.color_print('[', 'f_g_y')
        # number:多少个等号
        num3 = math.floor(index / self.length * self.bar_length)
        # ===>ooo]
        schedule = "" if num3 == 0 else (self.style[1] if num3 == 1 else self.style[0] * (num3 - 1) + self.style[1])
        remain = self.style[2] * (self.bar_length - num3)
        string += (self.color_print(schedule, 'f_g_g') + self.color_print(remain, 'f_g_r')
                   + self.color_print(']', 'f_g_y'))
        # Loss:0.15555 Accuracy:0.98 ...
        for value in message:
            num4 = str(value[1]) if value[2] is int else f"{value[1]:.{self.float_length}f}"
            string += self.color_print(value[0], 'f_g_p') + ":"
            string += self.color_print(num4, 'underline') + " "
        if index == self.length and self.newline:
            string += '\n'
        print(string, end='')


class LoadData:
    """💞
    function: 高光谱数据加载、预处理与划分工具类.该类用于读取指定数据集的图像和标签文件,完成按波段归一化、标签重编码、数据集信息统计、训练/测试样本划分,并可将数据
              转换为PyTorch Tensor供模型训练和测试使用.同时提供伪彩色可视化函数用于展示高光谱图像.
    """

    def __init__(self, data_name, data_path, is_print):
        """
        function : initialize==>(读取数据、归一化数据、统计并显示数据信息)
        data_name: 高光谱数据名称,可选项例如:[PaviaU,Salinas,...,Indian_pines]
        data_path: 高光谱数据地址.在该目录下应该存在两个数据文件,分别是图像数据文件、标签数据文件,例如:
                    data_path
                        |--image_data.mat
                        |--label_data.mat
        is_print : 是否打印数据信息
        """
        self.data_name = data_name
        self.data_path = data_path
        # 图像、标签、配置信息、类别个数
        image, label, config, number = self.read_data(data_name=data_name, data_path=data_path)
        # 显示数据信息,返回每个类别的数据个数
        class_number, channel = self.get_data_information(config=config, image=image, label=label, is_print=is_print)
        self.image = image
        self.label = label
        self.config = config
        self.number = number
        self.class_number = class_number  # 每个类别的个数(不包含背景像素)
        self.channel = channel
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def read_data(self, data_name, data_path):
        """
        function : 读取高光谱图像数据,并对高光谱数据归一化、将标签数据减一(背景标签变为-1)
        data_name: 数据集名称
        data_path: 数据集地址
        return   : [image、label、config]=[图像数据、标签数据、配置文件信息]
        Note     : image    ==>numpy数组,形状为(height,width,channel),图像数据已经归一化
                   label    ==>numpy数组,形状为(width,channel),标签数据已经减去1
                   config   ==>配置文件信息
                   num_class==>最大类别数
        """
        # 加载数据信息
        script_path = Path(__file__).resolve().parent  # 当前Python脚本的父目录
        with open(fr'{script_path}/KeyValuePairs.json', 'r', encoding='utf-8') as file:
            config = json.load(file)
        image_path = fr'{data_path}/{data_name}.mat'
        label_path = fr'{data_path}/{file_names[data_name]}.mat'
        image = sio.loadmat(image_path)[image_keys[data_name]].astype(np.float32)
        label = sio.loadmat(label_path)[label_keys[data_name]].astype(np.int32) - 1
        num_class = len(np.unique(label)) - 1
        image = self.norm_data(image=image)
        return image, label, config, num_class

    @staticmethod
    def norm_data(image: np.ndarray) -> np.ndarray:
        """
        function: 归一化数据:将数据归一化到[0,1]区间
        image   : 高光谱图像数据,numpy数组(height,width,channel)
        return  : 归一化后的数据
        """
        height, width, channel = image.shape
        image = image.reshape(height * width, channel)
        image = (image - np.min(image, 0)) / (np.max(image, 0) - np.min(image, 0) + 1e-8)
        return image.reshape(height, width, channel)

    def get_data_information(self, image, label, config, is_print=False):
        """
        function: 统计数据的信息,并显示在控制台
        image   : 图像数据(height,width,channel)
        label   : 标签数据(height,width),数据范围(-1~N-1)
        config  : 配置文件信息
        return  : class_number:[number1,...,numberN]==>每个类别的个数
                  channel     :输入数据的通道数
        """
        shape = image.shape  # 数据形状
        unique_label = np.unique(label)[1:]  # 类别标签
        valid_number = np.where(label != -1)[0].shape[0]  # 非背景像素个数
        valid_proportion = valid_number / (shape[0] * shape[1])
        class_proportion = []
        # 统计数据信息
        valid_proportion = f"{valid_proportion * 100:<5.2f}% number-{np.where(label != -1)[0].shape[0]}"
        message = {"Dataset name": self.data_name, "Data shape": f"{shape}",  # 数据形状
                   "Effective data proportion": valid_proportion,  # 有效数据占比
                   "Proportion of each category": None}  # 每个类别的数据占比
        class_number = []
        for i in unique_label:
            number = np.where(label == i)[0].shape[0]
            class_number.append(number)
            class_proportion.append(number / valid_number)
            float_number = f"{class_proportion[i] * 100:.2f}"
            message[f'label{i: >{len(str(len(unique_label)))}}:{config["data_label"][self.data_name][i]}'] \
                = f"{float_number: >5}% number-{number}"
        keys, values = message.keys(), message.values()
        max_key_length = max(len(s) for s in keys)
        red = "\033[91m"
        blue = "\033[94m"
        green = "\033[92m"
        reset = "\033[0m"
        if is_print:
            print("=" * 120 + "\n加载数据")
            print(f"{red}Data details{reset}")
            for key, value in zip(keys, values):
                if value is not None:
                    print(f"{green}{key: <{max_key_length}}{reset}:{blue}{value}{reset}")
                else:
                    print(f"{red}{key: <{max_key_length}}{reset}")
            print("=" * 120)
        return class_number, shape[2]

    def split_data(self, seeds, number: Union[int, List[int]], except_label: int, rate=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        function    : 将高光谱数据集划分为训练数据集、测试数据集
        seeds       : 随机种子
        number      : 训练数据中每个类别的个数
        except_label: 背景像素的标签值
        rate        : 按照比例取数据
        return      : 返回数据==>image_data,label_train,label_test
                      image_data  :[1,bands,height,width]数据类型:<class 'torch.Tensor'>
                      label_train :[1,height,width]      数据类型:<class 'torch.Tensor'>
                      label_test  :[1,height,width]      数据类型:<class 'torch.Tensor'>
                      label_attack:[1,height,width]      数据类型:<class 'torch.Tensor'>
        """
        np.random.seed(seeds)  # 设置随机种子
        number = number if type(number) is list else [number] * self.number if rate is None else [int(i * rate) for i in self.class_number]
        image_data = np.moveaxis(self.image, -1, 0)  # 形状由[height,width,bands]被改变为[bands,height,width]
        # 初始化两个空的数组存储训练集、测试集索引
        train_array, test_array = None, None
        for i in range(0, self.number):
            index = np.column_stack(np.where(self.label == i))
            np.random.shuffle(index)
            label_number = number[i]
            if i == 0:
                train_array = index[0:label_number, :]
                test_array = index[label_number:, :]
            else:
                train_array = np.concatenate((train_array, index[0:label_number, :]), axis=0)
                test_array = np.concatenate((test_array, index[label_number:, :]), axis=0)
        # 根据训练集、测试集索引获得训练、测试标签
        label_train, label_test = np.ones(self.label.shape) * except_label, np.ones(self.label.shape) * except_label,
        label_train[train_array[:, 0], train_array[:, 1]] = self.label[train_array[:, 0], train_array[:, 1]]
        label_test[test_array[:, 0], test_array[:, 1]] = self.label[test_array[:, 0], test_array[:, 1]]
        # 数据转换为张量
        image_data = torch.from_numpy(image_data).float().unsqueeze(0)
        label_train = torch.from_numpy(label_train).long().unsqueeze(0)
        label_test = torch.from_numpy(label_test).long().unsqueeze(0)
        return image_data, label_train, label_test

    @staticmethod
    def display_hsi_image(image_data, data_name, show=False, save_path=None):
        """
        function  : 二维可视化高光谱数据
        image_data: (height,width,bands) numpy数组
        data_name : 高光谱数据类型(Pavia、Data、...、Indian_pines)
        save_path : 高光谱图像可视化结果的保存目录(XXX/XXX/XXX),不包含文件名
        return    : None
        """
        set_matplotlib()
        rgb_band = {"PaviaU": [57, 34, 3], "Pavia": [57, 34, 3], "Indian_pines": [30, 20, 10], "Salinas": [43, 28, 16],
                    "Botswana": [104, 59, 29], "Houston13": [48, 27, 12], "Indian_pines_corrected": [30, 20, 10], "Salinas_corrected": [43, 28, 16]}
        if data_name not in rgb_band.keys():
            raise ValueError(f"数据集名称超出范围辣!!!目前仅支持以下数据集:{list(rgb_band.keys())}")
        # RGB波段数据
        band1 = image_data[:, :, rgb_band[data_name][0]]  # R
        band2 = image_data[:, :, rgb_band[data_name][1]]  # G
        band3 = image_data[:, :, rgb_band[data_name][2]]  # B
        pseudo_rgb = np.stack([band1, band2, band3], axis=-1)  # 将三个波段组合成一个RGB图像
        pseudo_rgb = pseudo_rgb / pseudo_rgb.max()  # 归一化数据到[0,1]
        pseudo_rgb_eq = np.zeros_like(pseudo_rgb)  # 应用直方图均衡化
        for i in range(pseudo_rgb.shape[-1]):
            pseudo_rgb_eq[:, :, i] = exposure.equalize_adapthist(pseudo_rgb[:, :, i])
        # 调整亮度
        brightness_factor = 1.5  # 调整亮度因子,大于1会使图像更亮
        pseudo_rgb_bright = pseudo_rgb_eq * brightness_factor
        # 应用Gamma校正
        gamma = 0.8  # Gamma值小于1会使图像更亮
        pseudo_rgb_gamma = np.power(pseudo_rgb_bright, gamma)
        pseudo_rgb_gamma = np.clip(pseudo_rgb_gamma, 0, 1)  # 确保图像值在[0,1]范围内
        # 显示图像
        plt.figure(figsize=(8, 8))
        plt.imshow(pseudo_rgb_gamma)
        plt.axis('off')
        if save_path is not None:
            plt.savefig(fr"{save_path}/{data_name}-FalseColor.png", dpi=300, bbox_inches='tight', pad_inches=0)
        if show:
            plt.title(f"{data_name}数据集假彩色图像")
            plt.show()


class CrossEntropy2d(nn.Module):
    """💞
    Description: 高光谱图像分类模型常用损失函数
    """

    def __init__(self, ignore_label, reduction='mean'):
        """
        function    : Initialize
        reduction   : 指定损失计算的归约方式==>控制损失函数的输出是每个样本的平均损失('mean')、
                      所有样本损失的总和('sum')、不进行任何归约('none')
        ignore_label: 指定在计算损失时忽略的标签值
        """
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict_label: torch.Tensor, true_label: torch.Tensor):
        """
        function: 自定义交叉熵损失函数,适用于处理二维图像分割任务中的像素级分类问题.
                  允许忽略特定的标签值,并选择性地对不同类别的损失进行加权,更灵活地适应不同的训练需求和数据特性
        predict : 模型的预测输出,形状为:[1,num_class,height,width]
        target  : 真实标签,形状为:[1,height,width]
        return  : loss
        """
        assert not true_label.requires_grad  # 确保target张量不需要梯度
        assert predict_label.dim() == 4  # predict张量的维度(dim)是4
        assert true_label.dim() == 3
        # 确保模型预测(predict)和目标标签(target)的相应维度相匹配.如果不相等==>抛出异常,显示两个维度的具体值
        assert predict_label.size(0) == true_label.size(0), "{0} vs {1} ".format(predict_label.size(0), true_label.size(0))
        assert predict_label.size(2) == true_label.size(1), "{0} vs {1} ".format(predict_label.size(2), true_label.size(1))
        assert predict_label.size(3) == true_label.size(2), "{0} vs {1} ".format(predict_label.size(3), true_label.size(3))
        batch_size, num_class, height, width = predict_label.size()
        target_mask = (true_label >= 0) * (true_label != self.ignore_label)  # target_mask.shape=[1,height,width]
        true_label = true_label[target_mask]  # 使用target_mask过滤target张量,只保留有效的标签值
        if not true_label.data.dim():  # 检查target张量是否有任何维度
            return torch.zeros(1)
        predict_label = predict_label.transpose(1, 2).transpose(2, 3).contiguous()  # predict.shape=[1,height,width,num_class]
        # 从predict张量中选择出对应于有效标签的预测值.predict.shape=[N,num_class]
        predict_label = predict_label[target_mask.view(batch_size, height, width, 1).repeat(1, 1, 1, num_class)].view(-1, num_class)
        # 计算交叉熵损失:若提供weight参数则对每个类别的损失进行加权,size_average用于控制损失的平均方式
        loss = func.cross_entropy(predict_label, true_label, weight=None, reduction=self.reduction)
        return loss


class Receiver:
    """💞
    function: 接收高光谱分类模型对其进行训练、测试、攻击、对抗样本测试
    """

    def __init__(self, model_name: str, save_path: str, data_name: str):
        """
        function  : Initialize
        model_name: 高光谱图像分类模型名称
        save_path : 运行结果保存的根目录
        dataset   : 数据集名称
        """
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ModelName = model_name
        self.RootDir = save_path
        self.DataSet = data_name
        self.LogPath = fr"{save_path}/Log.log"
        for file_name in ["Weight", "Json", "AdversarialExamples"]:
            file_dir = fr"{save_path}/{file_name}"
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
        # 预先创建类属性
        self.Model = None
        self.LearningRate = None
        self.Criterion = None
        self.Optimizer = None
        self.Height = None
        self.Width = None
        self.ExceptLabel = None
        self.AdjustLearningRate = None
        self.Epoch = None
        self.ModelType = None
        self.LabelMask = None
        self.LabelOneHot = None
        self.NumClass = None

    def reset(self, model: torch.nn.Module, except_label: int, settings: dict) -> None:
        """
        function    : 重置模型实例
        model       : 模型实例
        except_label: 忽略的标签值
        settings    : dict={"optimizer":AdamW,"lr":0.0005,"adjust_learning_rate":False,"weight_decay":0.00001,"loss_func":CrossEntropy2d()}
        return      : None
        """
        self.Model = model.to(self.Device)
        self.LearningRate = settings['lr']
        self.AdjustLearningRate = settings['adjust_learning_rate']
        self.ExceptLabel = except_label
        # 根据模型输出的数据类型选择合适的损失函数、优化器
        self.Criterion = settings['loss_func']
        self.Optimizer = settings['optimizer'](model.parameters(), lr=self.LearningRate, weight_decay=settings['weight_decay'])
        self.Epoch = settings['epoch']
        self.ModelType = settings["model_type"]
        self.NumClass = settings['num_class']

    @staticmethod
    def calculate_accuracy(predict_label: torch.tensor, true_label: torch.tensor, except_label: int, only_oa: bool = True) -> Union[float, Tuple[float, float, float, np.ndarray]]:
        """
        function     : 计算分类模型的准确率和其他统计指标(Kappa)
        predict_label: (1,num_class,height,width)==>预测结果是概率形式的
        true_label   : (1,height,width)
        except_label : 标记哪些数据(背景数据)不参与指标的计算
        only_oa      : 是否只计算总体准确率(OA)
        return       : 函数返回总体准确率(OA)、平均准确率、Kappa统计量和每个类别的准确率

        Note         : 必须保证:真实标签(true_label)中一定包含所有的类别
        """
        predict_label = torch.argmax(predict_label, dim=1)
        indexes = torch.argwhere(true_label[0] != except_label)
        # 取出预测标签以及真实标签
        predict_label = predict_label[0, indexes[:, 0], indexes[:, 1]]
        true_label = true_label[0, indexes[:, 0], indexes[:, 1]]
        # 计算总体准确率(Overall Accuracy)
        correct_number = (predict_label == true_label).sum().item()  # type:ignore # 预测正确的个数
        total_number = true_label.shape[0]  # 总数
        overall_accuracy = correct_number / total_number if total_number > 0 else 0
        if only_oa:
            return overall_accuracy
        # 计算混淆矩阵
        cm = confusion_matrix(true_label.cpu().numpy(), predict_label.cpu().numpy())
        # 计算每个类别的准确率(Class Accuracies)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        # 计算平均类别准确率(Average Accuracy)
        average_accuracy = np.nanmean(class_accuracies)
        # 计算Kappa系数
        kappa = cohen_kappa_score(true_label.cpu().numpy(), predict_label.cpu().numpy())
        return overall_accuracy, average_accuracy, kappa, class_accuracies

    def graph_label_process(self, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        function    :
        label       : (1,height,width),数据范围[0,...,num_class-1,except_label]
        return      : one_hot_label:[height*width,num_class]==>except_label对应的行全部为0,有效标签对应的行的对应标签位置数值为1.
                      label_mask   :[height*width,num_class]==>except_label对应的行全部为0,有效标签对应的行全部为1.
        """
        _, height, width = label.shape
        label_array = label.detach().cpu().numpy().reshape(-1)
        # raise KeyError
        # 有效像素
        valid_mask = label_array != self.ExceptLabel
        valid_labels = label_array[valid_mask]
        if np.any(valid_labels < 0) or np.any(valid_labels >= self.NumClass):  # 检查有效标签范围
            raise ValueError(f"除except_label={self.ExceptLabel}外,标签值必须位于[0,{self.NumClass - 1}]")
        pixel_num = label_array.size
        one_hot_label = np.zeros((pixel_num, self.NumClass), dtype=np.float32)  # 初始化One-Hot标签
        label_mask = np.zeros((pixel_num, self.NumClass), dtype=np.float32)  # 初始化标签掩码
        # 找出有效像素在展平数组中的位置
        valid_indices = np.flatnonzero(valid_mask)
        # 标签值直接作为类别列下标
        one_hot_label[valid_indices, label_array[valid_indices]] = 1.0
        # 有效像素对应的掩码整行设置为1
        label_mask[valid_indices, :] = 1.0
        return torch.from_numpy(one_hot_label).to(self.Device), torch.from_numpy(label_mask).to(self.Device)

    def parse_model_output(self, model_output: torch.Tensor, true_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        function    : 根据不同的模型,解析模型输出数据
        model_output: 模型输出
        true_label  : 真实标签(1,height,width)
        return      : 分类结果概率、损失值
        """
        if self.ModelType == "RCPNet":  # noqa
            predict_probability = model_output[0]
            loss = self.Criterion(predict_probability, model_output[1], true_label)
        elif self.ModelType == 'General':
            predict_probability = model_output
            loss = self.Criterion(predict_probability, true_label)
        elif self.ModelType == "BANet":
            predict_probability = model_output[0]
            loss = self.Criterion(predict_probability, true_label) + model_output[1]
        elif self.ModelType == "Graph":
            self.LabelOneHot, self.LabelMask = self.graph_label_process(label=true_label)
            predict_probability = model_output
            _, num_class, height, width = model_output.shape
            model_output = model_output.squeeze(0).permute(1, 2, 0).reshape(height * width, num_class)
            loss = self.Criterion(model_output, self.LabelOneHot, self.LabelMask)
        else:
            raise ValueError(f"不支持的模型输出类型:{self.ModelName},请在这个函数中添加相应的解析代码")
        return predict_probability, loss

    @staticmethod
    def get_epsilon_result(dict_data: dict, method='FGSM', epsilon: float = 0.04) -> list:
        """
        function : 从对抗样本的测试结果中取出指定攻击方法(method)和扰动幅度(epsilon)的测试结果
        dict_data: {
                    '1': {"PGD-0.04": [Class1,...,ClassN,OA,AA,Kappa], "CW-0.04": [Class1,...,ClassN,OA,AA,Kappa], "method-epsilon":[...],...,},
                    '2': {"PGD-0.04": [Class1,...,ClassN,OA,AA,Kappa], "CW-0.04": [Class1,...,ClassN,OA,AA,Kappa], "method-epsilon":[...],...,},
                    ...
                    'M': {"PGD-0.04": [Class1,...,ClassN,OA,AA,Kappa], "CW-0.04": [Class1,...,ClassN,OA,AA,Kappa], "method-epsilon":[...],...,},
                   }
        method   : 攻击方法
        epsilon  : 扰动幅度
        return   : [[Class1,...,ClassN,OA,AA,Kappa],...,[Class1,...,ClassN,OA,AA,Kappa]]

        Note     : 其中M是训练次数、N是类别个数,返回的列表长度为M
        """
        if method not in ["FGSM", "PGD", "CW"]:
            raise ValueError(fr"尚不支持的攻击方法==>{method}!!!")
        result = []
        key = method + '-' + str(epsilon)
        for keys in dict_data.keys():
            value_data = dict_data[keys][key]
            result.append(value_data)
        return result

    @staticmethod
    def calculate_mean_std(result: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        function: 计算所有结果的均值和标准差
        result  : result=[[value1,...,valueN],[value1,...,valueN]]
        return  : 均值、方差,形状均为(N,1)

        Note    : result中的数据均是位于0~1之间的浮点数
        """

        def process_float_number(array):
            """
            function:将一维数组的每个元素截断到两位小数,不进行四舍五入
            """
            return np.floor(array * 100) / 100

        data = np.array(result) * 100
        m, n = data.shape
        means = process_float_number(np.mean(data, axis=0))
        stds = np.zeros(n) if m == 1 else np.std(data, axis=0, ddof=1)
        return means, stds

    def convert_result_to_text(self, result: Union[Dict, List], mode, column_number=4) -> str:
        """
        function     : 将结果转换为可直接打印的文本
        result       : 程序运行结果
        mode         : ['train','test','attack','summary','contrast']
        column_number: 显示的列数
        return       : text
        """
        if mode not in ['train', 'test', 'attack', 'summary', 'contrast']:
            raise ValueError("模式写错辣!!!")
        text = ""
        if mode == "train":  # 数据的格式为:{'OA': float,'Loss': float}
            for key, value in result.items():
                text += f"{key}:{value :8.8f}    " if key == "Loss" else f"{key}:{value * 100 :8.4f}%    "
            text += "\n"
        elif mode == 'test':  # 数据的格式为:{'OA': float, 'AA': float, 'Kappa': float, 'Loss': float', 'Class_Accuracy': list}
            ls = len(str(len(result["Class_Accuracy"]))) + 6
            oa, aa, kappa, loss, = result['OA'] * 100, result['AA'] * 100, result['Kappa'] * 100, result['Loss']
            accuracies = result['Class_Accuracy']
            text += f"{'OA':<{ls}}:{oa:8.4f}%    {'AA':<{ls}}:{aa:8.4f}%    {'Kappa':<{ls}}:{kappa:8.4f}%    {'Loss':<{ls}}:{loss:8.7f}\n"
            for index, accuracy in enumerate(accuracies):
                text += f"class-{index:>{ls - 6}}:{accuracy * 100:>8.4f}%" + "    "
                if (index + 1) % column_number == 0:
                    text += "\n"
        elif mode == 'attack':
            pass
        elif mode == 'summary':  # 数据的格式为:[[Class1,...,ClassN,OA,AA,Kappa], ..., [Class1,...,ClassN,OA,AA,Kappa]]
            means, stds = self.calculate_mean_std(result)
            ls = max(2, len(str(means.shape[0] - 3)))
            for i in range(means.shape[0] - 3):
                text += f"{i: >{ls}}:{means[i]:6.2f}±{stds[i]:.2f}\n"
            text += f"OA:{means[-3]:6.2f}%±{stds[-3]:.2f} AA:{means[-2]:6.2f}%±{stds[-2]:.2f} Kappa:{means[-1]:6.2f}%±{stds[-1]:.2f}\n"
        elif mode == 'contrast':  # 数据的格式为{'attack_method-epsilon':[[Class1,...,ClassN,OA,AA,Kappa], ..., [Class1,...,ClassN,OA,AA,Kappa]],...,}
            text += "*" * 100 + '\n'
            for key, value in result.items():
                method, epsilon = key.split('-')
                means, stds = self.calculate_mean_std(value)
                text += f"在攻击方法⇒{method},扰动幅度⇒{epsilon}下,各个指标数值为:\n"
                text += f"0A:{means[-3]:6.2f}%±{stds[-3]:.2f} AA:{means[-2]:6.2f}%±{stds[-2]:.2f} Kappa:{means[-1]:6.2f}%±{stds[-1]:.2f}\n"
            text += "*" * 100
        return text

    def record_result_to_log(self, description: str, result, mode: str, is_print) -> None:
        """
        function   : 将计算结果保存至相应的日志文件
        description: 关于结果的描述
        result     : 运行结果
        mode       : ['train','test','attack']
        return     : None
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = "\n" + current_time + description + '\n' + self.convert_result_to_text(result=result, mode=mode)
        if is_print:
            print(text)
        with open(self.LogPath, 'a', encoding='utf-8') as file:
            file.write(text)

    def adjust_learning_rate(self, iteration, power=0.9):
        lr = self.LearningRate * ((1 - float(iteration) / self.Epoch) ** power)
        self.Optimizer.param_groups[0]['lr'] = lr
        if len(self.Optimizer.param_groups) > 1:
            self.Optimizer.param_groups[1]['lr'] = lr * 10

    def train_model(self, train_image: torch.tensor, train_label: torch.tensor, index: int = 1, is_save: bool = True, show_time=True) -> None:
        """
        function   : 训练模型
        train_image: (1,band,height,width)==>图像数据
        train_label: (1,height,width)     ==>训练标签数据
        index      : 模型索引(这是第几个训练的模型)
        is_save    : 是否保存模型权重
        show_time  : 是否显示平均每轮训练时间
        return     : None
        """
        # 数据转移至设备
        train_image = train_image.to(self.Device)
        train_label = train_label.to(self.Device)
        self.Model.train()
        # 训练模型
        print(f"\033[92m数据集{self.DataSet}-第{index}次训练模型{self.ModelName}:\033[0m")
        bar = ColorProgressBar(length=self.Epoch, description="Epoch", bar_length=40)
        train_accuracy, loss_value = 0, 0
        time1 = time.time()
        for epoch in range(self.Epoch):
            if self.AdjustLearningRate:
                self.adjust_learning_rate(iteration=epoch)
            self.Optimizer.zero_grad()
            output = self.Model(train_image)
            # 解析模型输出
            predict_probability, loss = self.parse_model_output(model_output=output, true_label=train_label)
            loss_value = loss.item()
            loss.backward()
            self.Optimizer.step()
            train_accuracy = self.calculate_accuracy(predict_probability, train_label, self.ExceptLabel, only_oa=True)
            message = [["Loss", loss_value, float], ["Accuracy", train_accuracy, float]]
            bar.show_progress_bar(index=epoch + 1, message=message)
        time2 = time.time()
        if show_time:
            print(f"平均每轮训练时间为:\033[34m{(time2 - time1) / self.Epoch}\033[0m/epoch")
        # 保存模型
        if is_save:
            save_root_path = fr"{self.RootDir}/Weight/{self.ModelName}/{self.DataSet}"
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)
            torch.save(self.Model, fr'{save_root_path}/{index}.pt')
        # 保存训练结果至日志文件
        result = {"OA": train_accuracy, "Loss": loss_value}
        description = f" 模型{self.ModelName}在数据集{self.DataSet}上进行第{index}次训练:\n训练结果如下:"
        self.record_result_to_log(description=description, result=result, mode='train', is_print=True)

    def against_train_model(self, train_image: torch.tensor, train_label: torch.tensor, epochs, epsilon, index, is_save) -> None:
        train_image = train_image.to(self.Device)
        train_label = train_label.to(self.Device)
        self.Model.train()
        # 训练模型
        print(f"\033[92m模型{self.ModelName}进行对抗训练\033[0m")
        bar = ColorProgressBar(length=epochs, description="Epoch", bar_length=40)
        for epoch in range(epochs):
            self.Optimizer.zero_grad()
            adv_image = self.target_fgsm_with_norm(image=train_image, label=train_label, epsilons=[epsilon], root_dir='', show_bar=False)
            adv_image = torch.from_numpy(adv_image[f'FGSM-{epsilon}']).unsqueeze(0).to(self.Device)
            against_train_output = self.Model(adv_image)
            against_train_probability, against_train_loss = self.parse_model_output(model_output=against_train_output, true_label=train_label)
            against_accuracy = self.calculate_accuracy(against_train_probability, train_label, self.ExceptLabel, only_oa=True)
            loss_value = against_train_loss.item()
            self.Optimizer.zero_grad()
            against_train_loss.backward()
            self.Optimizer.step()
            message = [["AgainstTrainAccuracy", against_accuracy, float], ['AgainstTrainLoss', loss_value, float]]
            bar.show_progress_bar(index=epoch + 1, message=message)
        if is_save:
            save_root_path = fr"{self.RootDir}/Weight/{self.ModelName}/{self.DataSet}"
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)
            torch.save(self.Model, fr'{save_root_path}/{index}.pt')

    def test_model(self, test_image, test_label, index=1, show_time=True):
        """
        function    : 测试模型
        test_image  : 图像数据(1,band,height,width),数据类型:torch.tensor
        test_label  : 测试标签数据(1,height,width),数据类型:torch.tensor
        return      : [Class1,...,ClassN,OA,AA,Kappa]
        show_time   : 是否显示平均每轮训练时间
        """
        print(f"\033[92m测试模型:\033[0m", end="")
        self.Model.eval()
        test_image = test_image.to(self.Device)
        test_label = test_label.to(self.Device)
        self.Model.to(self.Device)
        time1 = time.time()
        output = self.Model(test_image)
        time2 = time.time()
        if show_time:
            print(f"平均每轮测试时间为:\033[34m{(time2 - time1)}\033[0m/epoch")
        predict_probability, loss = self.parse_model_output(model_output=output, true_label=test_label)
        loss_value = loss.item()
        oa, aa, kappa, class_accuracies = self.calculate_accuracy(predict_probability, test_label, self.ExceptLabel, False)
        result = {"OA": oa, "AA": aa, "Kappa": kappa, "Loss": loss_value, "Class_Accuracy": class_accuracies}
        description = f" 模型{self.ModelName}在数据集{self.DataSet}上进行第{index}次测试:\n测试结果如下:"
        self.record_result_to_log(description=description, result=result, mode='test', is_print=True)
        return *class_accuracies, oa, aa, kappa

    def target_fgsm_with_norm(self, image, label, root_dir, epsilons, target_value=0, is_save=False, show_bar=True) -> dict:
        """
        function    : 带有范数的定向快速梯度符号攻击
        model       : 被攻击的模型
        image       : 图像数据
        label       : 真实标签
        root_dir    : 对抗样本、噪声数据的所属根目录
        epsilons    : 扰动级别
        target_value: 定向攻击的标签值
        is_save     : 是否保存对抗样本、对抗性噪声
        show_bar    : 是否显示进度条
        return      : 对抗样本数据

        Note        : return_data={"FGSM-0":numpy.array,...,"FGSM-0.05":numpy.array}
        """
        length = max(len(str(e)) for e in epsilons) + 1
        bar = ColorProgressBar(length=len(epsilons), description=" ", bar_length=10)
        self.Model.eval()
        target_label = (torch.ones_like(label) * target_value).long().to(self.Device)  # 定向标签
        adversarial_example = {}
        for index, epsilon in enumerate(epsilons):
            bar.description = f"攻击模型-FGSM-Epsilon:{str(epsilon):>{length}}"  # noqa
            processed_image = Variable(image).to(self.Device)
            processed_image = processed_image.requires_grad_()
            output = self.Model(processed_image)
            _, loss = self.parse_model_output(model_output=output, true_label=target_label)
            loss.backward()
            perturbation = epsilon * processed_image.grad.data / torch.norm(processed_image.grad.data, float("inf"))
            noise = perturbation.cpu().data.numpy()[0]
            processed_image.data = processed_image.data - perturbation
            adv_image = torch.clamp(processed_image, 0, 1).cpu().data.numpy()[0]
            adversarial_example[f"FGSM-{epsilon}"] = adv_image  # noqa
            if is_save:
                adv_path = root_dir + fr"/FGSM-Adv-{epsilon}.mat"  # noqa
                noise_path = root_dir + fr"/FGSM-noise-{epsilon}.mat"  # noqa
                sio.savemat(adv_path, {"image": adv_image})
                sio.savemat(noise_path, {"image": noise})
            if show_bar:
                bar.show_progress_bar(index=index + 1, message=[])
        return adversarial_example

    def pgd_attack(self, image, true_label, root_dir, epsilons, iteration=10, is_save=False, target_value=0):
        """
        function    : (Projected Gradient Descent)攻击
        model       : 被攻击的模型
        image       : 图像数据
        true_label  : 真实标签
        root_dir    : 对抗样本、噪声数据的所属根目录
        epsilons    : 扰动级别
        iteration   : 迭代次数
        is_save     : 是否保存对抗样本、对抗性噪声
        target_value: 定向攻击的标签值
        return      : 对抗样本数据

        Note        : return_data={"PGD-0.05":numpy.array,...,"PGD-0.1":numpy.array}
        """
        self.Model.eval()
        result_dict = {}
        length = max(len(str(e)) for e in epsilons) + 1
        bar = ColorProgressBar(length=len(epsilons), description=" ", bar_length=10)
        for index, epsilon in enumerate(epsilons):
            bar.description = f"攻击模型-PGD -Epsilon:{str(epsilon):>{length}}"
            pgd_image = image.clone().to(self.Device)
            pgd_label = (torch.ones_like(true_label) * target_value).long().to(self.Device)
            for _ in range(iteration):
                pgd_image = Variable(pgd_image).cuda()
                pgd_image.requires_grad_()
                output = self.Model(pgd_image)
                _, loss = self.parse_model_output(model_output=output, true_label=pgd_label)
                self.Model.zero_grad()
                loss.backward()
                grad = pgd_image.grad.data
                perturbation = epsilon * grad / torch.norm(grad, p=float("inf"))
                pgd_image = pgd_image.detach() - perturbation
                pgd_image = torch.clamp(pgd_image, 0, 1)
            bar.show_progress_bar(index=index + 1, message=[])  # noqa
            pgd_image_array = pgd_image.cpu().data.numpy()[0]
            if is_save:
                adv_path = root_dir + fr"/PGD-Adv-{epsilon}.mat"
                sio.savemat(adv_path, {"image": pgd_image_array})
            result_dict[f"PGD-{epsilon}"] = pgd_image_array
        return result_dict

    def cw_attack(self, image, true_label, root_dir, epsilons, iteration=10, C=50, is_save=False):  # noqa
        """
        function  : C&W攻击
        model     : 被攻击的模型
        image     : 图像数据
        true_label: 真实标签
        root_dir  : 对抗样本、噪声数据的所属根目录
        epsilons  : 扰动级别
        iteration : 迭代次数
        C         : 控制攻击成功率与扰动幅度之间的平衡
        is_save   : 是否保存对抗样本、对抗性噪声
        return    : return_data={"CW-0":numpy.array,...,"CW-0.05":numpy.array}
        """
        self.Model.eval()
        result_dict = {}
        length = max(len(str(e)) for e in epsilons) + 1
        num_class = torch.unique(true_label).shape[0] - 1
        target_label = torch.zeros_like(true_label).long().to(self.Device)
        one_hot_label = func.one_hot(target_label, num_classes=num_class)
        one_hot_label = torch.moveaxis(one_hot_label, -1, 1).to(self.Device)
        bar = ColorProgressBar(length=len(epsilons), description=" ", bar_length=10)
        for index, epsilon in enumerate(epsilons):
            bar.description = f"攻击模型-C&W -Epsilon:{str(epsilon):>{length}}"
            cw_image = image.clone().to(self.Device)
            for _ in range(iteration):
                cw_image = Variable(cw_image).requires_grad_(True)
                output = self.Model(cw_image)
                predict_probability, _ = self.parse_model_output(model_output=output, true_label=target_label)
                correct_logit = torch.mean(torch.sum(one_hot_label * predict_probability, dim=1))
                wrong_logit = torch.mean(torch.max((1 - one_hot_label) * predict_probability, dim=1)[0])
                loss = correct_logit - wrong_logit + C
                self.Model.zero_grad()
                loss.backward()
                grad = cw_image.grad.data
                perturbation = epsilon * grad / torch.norm(grad, p=float("inf"))
                cw_image = cw_image.detach() + perturbation
                cw_image = torch.clamp(cw_image, 0, 1)
            bar.show_progress_bar(index=index + 1, message=[])  # noqa
            cw_image_array = cw_image.cpu().data.numpy()[0]
            if is_save:
                adv_path = root_dir + fr"/CW-Adv-{epsilon}.mat"
                sio.savemat(adv_path, {"image": cw_image_array})
            result_dict[f"CW-{epsilon}"] = cw_image_array
        return result_dict

    def attack_model(self, image, true_label, except_label, model_index=1, is_save=False, pgd_epsilons=None, cw_epsilons=None, fgsm_epsilons=None) -> dict:  # noqa
        """
        function    : 攻击模型,返回对抗样本数据
        image       : 图像数据
        true_label  : 真实标签
        except_label: 背景标签
        model_index : 攻击次数
        is_save     : 是否保存对抗样本、噪声数据.
        return      : {
                       'method-epsilon': [Class1,Class2,...,ClassN,OA,AA,Kappa],
                       ......,
                       'method-epsilon': [Class1,Class2,...,ClassN,OA,AA,Kappa],
                      }
        """
        # 设置对抗样本、对抗性噪声的保存地址
        save_root_dir = self.RootDir + fr"/AdversarialExamples/{self.ModelName}/{self.DataSet}/{model_index}"
        if not os.path.exists(save_root_dir):
            os.makedirs(save_root_dir)
        print(f"\033[92m数据集{self.DataSet}-第{model_index}次攻击模型{self.ModelName}:\033[0m")
        self.Model.eval()
        fgsm_adv_image, pgd_adv_image, cw_adv_image = None, None, None
        # 攻击模型-(Fast Gradient Sign Method)
        if fgsm_epsilons is not None:
            fgsm_adv_image = self.target_fgsm_with_norm(image, true_label, save_root_dir, fgsm_epsilons, 0, is_save)
        # 攻击模型-(Projected Gradient Descent)
        if pgd_epsilons is not None:
            pgd_adv_image = self.pgd_attack(image, true_label, save_root_dir, pgd_epsilons, is_save=is_save)
        # 攻击模型-(C&W)
        if cw_epsilons is not None:
            cw_adv_image = self.cw_attack(image, true_label, save_root_dir, cw_epsilons, is_save=is_save)
        # 测试对抗样本
        print(f"测试对抗样本:")
        with torch.no_grad():
            adv_image = fgsm_adv_image | pgd_adv_image | cw_adv_image
            all_result = {}
            true_label = true_label.to(self.Device)
            for methods in ["PGD", "CW", "FGSM"]:
                test_epsilon = None
                if methods == "FGSM":
                    test_epsilon = fgsm_epsilons
                elif methods == "PGD":
                    test_epsilon = pgd_epsilons
                elif methods == "CW":
                    test_epsilon = cw_epsilons
                if test_epsilon is None:
                    continue
                length = max(len(str(e)) for e in test_epsilon) + 1
                print(f"攻击方法-{methods}:")
                result = ""
                for index, e in enumerate(test_epsilon):
                    image = adv_image[f"{methods}-{e}"]
                    image = torch.from_numpy(image).float().unsqueeze(0).to(self.Device)
                    output = self.Model(image)
                    probability, loss = self.parse_model_output(model_output=output, true_label=true_label)
                    loss_value = loss.item()
                    oa, aa, kappa, class_accuracies = self.calculate_accuracy(probability, true_label, except_label, False)
                    all_result[f"{methods}-{e}"] = [*class_accuracies, oa, aa, kappa]
                    text = f"Epsilon:{str(e):>{length}}   OA:{oa * 100:8.4f}%  AA:{aa * 100:8.4f}% Kappa:{kappa * 100:8.4f}% "f"Loss:{loss_value:8.7f}"
                    result += text + "\n"
                    print(text)
                with open(self.LogPath, 'a', encoding='utf-8') as file:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file.write(
                        current_time + f" 模型{self.ModelName}在数据集{self.DataSet}上进行第{model_index}次对抗样本测试,攻击方法为{methods}:\n对抗样本测试结果如下:\n")
                    file.write(result)
                torch.cuda.empty_cache()
        return all_result


class SLIC(object):
    """💞
    Description: 基于SLIC算法对输入图像进行超像素分割,并构建像素-超像素映射矩阵、超像素特征矩阵以及超像素邻接矩阵.
    """

    def __init__(self, image_data, label_data, nun_segments=1000, compactness=20, max_iter=20, sigma=0,
                 min_size_factor=0.3, max_size_factor=2):
        """
        function       : Initialize
        image_data     : (height,width,channel)
        label_data     : (height,width)
        nun_segments   : 目标超像素的数量
        compactness    : 控制超像素紧凑度
        max_iter       : 最大迭代次数
        sigma          : 用于平滑图像的高斯核的标准差
        min_size_factor: 控制超像素的最小尺寸
        max_size_factor: 控制超像素的最大尺寸
        """
        self.num_segment = nun_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        height, width, bands = image_data.shape
        flatten_data = np.reshape(image_data, [height * width, bands])
        normed_data = preprocessing.StandardScaler()  # 将数据的特征列进行标准化,使其均值为0,标准差为1
        flatten_data = normed_data.fit_transform(flatten_data)
        self.data = np.reshape(flatten_data, [height, width, bands])
        self.labels = label_data
        self.segments = None  # 分割结果
        self.superpixel_count = None  # 超像素个数
        self.feature_matrix = None  # 特征矩阵==>超像素的特征向量
        self.map_matrix = None  # 映射矩阵==>每个像素属于哪个超像素

    def segment_data(self):
        """
        function: 对图像进行超像素分割
        return  : map_matrix    : (height*width,super_pixel_number)映射矩阵==>每个像素属于哪个超像素(one-hot标记)
                  feature_matrix: (super_pixel_number,channel)==>特征矩阵
                  segments      : (height,width)==>分割结果(height,width)==>每个元素值表示该像素属于哪个超像素
        """
        image = self.data
        height, width, channel = image.shape
        segments = slic(image, n_segments=self.num_segment, compactness=self.compactness, max_num_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False,
                        start_label=0)  # 分割结果(height,width)==>每个元素值表示该像素属于哪个超像素
        # 如果分割结果中的最大值加1不等于超像素的数量,调用SegmentsLabelProcess()函数进行处理
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = self.segment_label_process(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        segments = np.reshape(segments, [-1])  # (height*width,1)
        feature_matrix = np.zeros([superpixel_count, channel], dtype=np.float32)  # 特征矩阵==>超像素的特征向量
        map_matrix = np.zeros([width * height, superpixel_count], dtype=np.float32)  # 映射矩阵==>每个像素属于哪个超像素
        flatten_image = np.reshape(image, [-1, channel])  # (height*width,channel)
        for i in range(superpixel_count):
            index = np.where(segments == i)[0]
            count = len(index)
            pixels = flatten_image[index]
            superpixel = np.sum(pixels, 0) / count
            feature_matrix[i] = superpixel
            map_matrix[index, i] = 1
        self.feature_matrix = feature_matrix
        self.map_matrix = map_matrix
        return map_matrix, feature_matrix, self.segments

    @staticmethod
    def segment_label_process(labels: numpy.array):
        """💞
        function: 对输入的标签数组进行重新编号,使得每个唯一的标签值被映射到一个从0开始的连续整数
        labels  : (height,width)
        return  : (height,width)
        Note    : 10,20,10      0,1,0
                  20,30,20 ===> 1,2,1
                  10,20,30      0,1,2
        """
        labels = np.array(labels, np.int64)
        height, width = labels.shape
        ls = list(set(np.reshape(labels, [-1]).tolist()))  # 提取所有唯一标签值
        dic = {}
        for i in range(len(ls)):
            dic[ls[i]] = i
        new_labels = labels
        for i in range(height):
            for j in range(width):
                new_labels[i, j] = dic[new_labels[i, j]]
        return new_labels

    def get_adjacency_matrix(self, sigma: float):
        """
        function: 计算超像素之间的邻接矩阵
        sigma   : 控制相似度计算中的高斯核的标准差
        return  : (super_pixel_number,super_pixel_number)
        """
        adjacency_matrix = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        height, width = self.segments.shape
        for i in range(height - 2):
            for j in range(width - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if adjacency_matrix[idx1, idx2] != 0:
                        continue
                    pix1 = self.feature_matrix[idx1]
                    pix2 = self.feature_matrix[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    adjacency_matrix[idx1, idx2] = adjacency_matrix[idx2, idx1] = diss
        return adjacency_matrix


class LdaSlic(object):
    """💞
    Description: 使用SLIC算法生成超像素分割结果,并提取超像素级特征与邻接关系.
    """

    def __init__(self, image, label, pca_dimension):
        """
        function     : Initialize
        image        : (height,width,channel)
        label        : (height,width)
        pca_dimension: LDA降维后的目标维度数
        """
        self.data = image
        self.init_labels = label
        self.curr_data = image
        self.n_component = pca_dimension
        self.height, self.width, self.bands = image.shape
        self.flatten_image = np.reshape(image, [self.width * self.height, self.bands])
        self.flatten_label = np.reshape(label, [self.height * self.width])
        self.label = label

    def linear_discriminant_analysis(self, curr_labels):
        """
        function   : 对输入的图像数据进行线性判别分析:LDA降维
        curr_labels: (height,width)
        return     : (height,width,n_component)==>降维后的数据
        """
        curr_labels = np.reshape(curr_labels, [-1])  # (height,width)==>(height*width,)
        index = np.where(curr_labels != 0)[0]
        x = self.flatten_image[index]
        y = curr_labels[index]
        lda = LinearDiscriminantAnalysis()
        lda.fit(x, y - 1)
        new_data = lda.transform(self.flatten_image)  # (height*width,n_component)
        return np.reshape(new_data, [self.height, self.width, -1])

    def simple_linear_iterative_clustering(self, input_data, scale=25):
        """
        function  : 对图像数据进行超像素分割
        input_data: (height,width,channel')降维后的图像数据
        scale     : 控制超像素分割的粒度
        return    : map_matrix      : 映射矩阵==>(height*width,K)每个像素属于哪个超像素(one-hot标记).若map_matrix[p,k]=1,则第p个像素属于第k个超像素.
                    feature_matrix  : 特征矩阵==>(K,channel)每个超像素的平均特征.
                    adjacency_matrix: 邻接矩阵==>(K,K),表示超像素之间的邻接关系,不是简单的0/1邻接矩阵,而是带权重的相似度矩阵.
                    segment         : 超像素分割结果==>(height,width),表示每个像素属于哪个超像素,segment[10,20]=5,表示第10行、第20列的像素属于第5个超像素.超像素编号是(0~K-1).

        """
        segments_init = self.height * self.width / scale  # 计算初始超像素的数量
        segment_func = SLIC(input_data, label_data=self.label, nun_segments=segments_init, compactness=1, sigma=1,
                            min_size_factor=0.1, max_size_factor=2)
        map_matrix, feature_matrix, segment = segment_func.segment_data()
        adjacency_matrix = segment_func.get_adjacency_matrix(sigma=10)
        return map_matrix, feature_matrix, adjacency_matrix, segment

    def simple_superpixel(self, scale):
        """
        function: 使用初始标签对图像数据进行LDA降维、使用降维后的数据进行SLIC超像素分割、提取超像素的特征矩阵、相似度矩阵和邻接矩阵、
                  返回超像素分割结果和相关特征矩阵
        scale   : 控制超像素分割的粒度
        return  : map_matrix      : 映射矩阵==>每个像素属于哪个超像素(one-hot标记)
                  feature_matrix  : 特征矩阵
                  adjacency_matrix: 邻接矩阵,表示超像素之间的邻接关系
                  segment         : 超像素分割结果,表示每个像素属于哪个超像素
        """
        curr_labels = self.init_labels
        lda_data = self.linear_discriminant_analysis(curr_labels)  # 降维后的数据
        map_matrix, feature_matrix, adjacency_matrix, segment = self.simple_linear_iterative_clustering(lda_data, scale)
        return map_matrix, feature_matrix, adjacency_matrix, segment


class AnalysisModel:
    """💞
    Description: 用于分析模型内部特征、BatchNorm统计参数以及反向传播梯度响应.该类通过前向钩子和后向钩子获取指定目标层的输入/输出特征、通道级均值与方差、
                 梯度分布等信息,从而辅助分析模型在不同输入条件下的特征表达和统计变化.
    """

    def __init__(self, weight_path: str, except_label=255):
        """
        function    : Initialize
        weight_path : 模型权重地址文件
        except_label: 背景标签的指定数值
        """
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Model = torch.load(weight_path, weights_only=False).to(self.Device)
        self.Criterion = CrossEntropy2d(ignore_label=except_label).to(self.Device)
        self.ExceptLabel = except_label

    @staticmethod
    def get_feature_mean_var_hook(cache: dict) -> Callable:
        """
        function   : 返回前向钩子函数(返回输入、输出特征的均值、方差数据)
        cache      : 保存输出特征的均值、方差
        return     : 前向钩子函数
        """

        def hook(module, inputs, output) -> None:  # noqa
            """
            function: 前向钩子函数
            module  : 选中的模块
            inputs  : 输入特征
            output  : 输出特征
            return  : None
            """
            input_datas = inputs[0].detach().cpu()
            input_means = input_datas.mean(dim=(0, 2, 3), keepdim=True).view(-1, 1).numpy().squeeze(1)  # noqa
            input_vars = input_datas.var(dim=(0, 2, 3), keepdim=True, unbiased=False).view(-1, 1).numpy().squeeze(1)
            output_datas = output.detach().cpu()
            output_means = output_datas.mean(dim=(0, 2, 3), keepdim=True).view(-1, 1).numpy().squeeze(1)  # noqa
            output_vars = output_datas.var(dim=(0, 2, 3), keepdim=True, unbiased=False).view(-1, 1).numpy().squeeze(1)
            cache["input_mean"] = input_means
            cache["input_var"] = input_vars
            cache["output_mean"] = output_means
            cache["output_var"] = output_vars

        return hook

    @staticmethod
    def get_feature_hook(cache: dict) -> Callable:
        """
        function   : 返回前向钩子函数(返回输入、输出特征的数据)
        cache      : 保存输入、输出特征的均值、方差
        return     : 前向钩子函数
        """

        def hook(module, inputs, output) -> None:  # noqa
            """
            function: 前向钩子函数
            module  : 选中的模块
            inputs  : 输入特征
            output  : 输出特征
            return  : None
            """
            # output_datas = output.detach().cpu()
            input_datas = inputs[0].detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
            output_datas = output.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
            cache["input"] = input_datas
            cache["output"] = output_datas

        return hook

    @staticmethod
    def get_gradient_hook(cache: dict) -> Callable:
        """
        function   : 返回后向钩子函数
        cache      : 保存输入、输出特征的均值、方差
        return     : 后向钩子函数
        """

        def hook(module, grad_input, grad_output) -> None:  # noqa
            """
            function  : 后向钩子函数,返回归一化后的梯度(输入、输出数据的梯度)
            grad_input: 输入特征的梯度
            rad_output: 输出特征的梯度
            return    : None
            """
            grad_in = grad_input[0].detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
            grad_out = grad_output[0].detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
            grad_in = LoadData.norm_data(grad_in)
            grad_out = LoadData.norm_data(grad_out)
            cache['grad_input'] = grad_in
            cache['grad_output'] = grad_out

        return hook

    def parsing_module(self, target_layer: str) -> nn.Module:
        """
        function    : 从模型中解析出目标模块
        target_layer: 目标模块
        return      : 目标模块
        """
        current_layer = self.Model
        pattern = r"(\w+|\[\d+\])"
        for module in re.findall(pattern, target_layer):
            if module[0] == '[' and module[-1] == ']':
                current_layer = current_layer[int(module[1:-1])]
            else:
                if not hasattr(current_layer, module):
                    raise AttributeError(f"Module has no attribute {module}")
                current_layer = getattr(current_layer, module)
        return current_layer

    def get_batch_norm_params(self, target_layer) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        function    : 获取BatchNorm的Running_mean、Running_var数值
        target_layer: BatchNorm模块名称
        return      : Running_mean、Running_var
        """
        batch_norm = self.parsing_module(target_layer=target_layer)
        gama_parm = batch_norm.weight.data.cpu().numpy()
        beta_parm = batch_norm.bias.data.cpu().numpy()
        running_mean = batch_norm.running_mean.cpu().numpy()
        running_var = batch_norm.running_var.cpu().numpy()
        return running_mean, running_var, gama_parm, beta_parm

    def get_feature_mean_var(self, target_layer: str, adv_image, mode) -> Tuple[np.ndarray, np.ndarray]:
        """
        function    : 获取输入目标层的特征的均值、方差数据
        target_layer: 目标层
        adv_image   : 图像数据(channel,height,width)
        mode        :
        return      : mean==>np.array=[C,]
                      var ==>np.array=[C,]
        """
        if mode not in ['input', 'output']:
            raise ValueError("模式写错辣!!!")
        # 缓存字典,保存需要的结果
        mean_var_cache = {}
        # 解析模块
        layer = self.parsing_module(target_layer=target_layer)
        # 加载图像数据
        image_data = torch.from_numpy(adv_image).unsqueeze(0).to(self.Device)
        # 加载标签数据
        forward_hook = layer.register_forward_hook(self.get_feature_mean_var_hook(cache=mean_var_cache))
        self.Model.eval()
        _ = self.Model(image_data)
        forward_hook.remove()
        if mode == "input":
            return mean_var_cache['input_mean'], mean_var_cache['input_var']
        else:
            return mean_var_cache['output_mean'], mean_var_cache['output_var']

    def get_gradients(self, image_message: dict, target_layer: str, target_value=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        function     : 获取模型在反向传播中模块(target_layer)的输入、输出数据的梯度,梯度已经归一化
        image_message: 图像数据地址dict={'path': .mat数据文件地址,'key':数据键值}
        target_layer : 目标模块
        target_value : 定向攻击的指定标签
        return       : input_data.gradient、output_data.gradient形状均为(height,width,channel)

        Note         : (1)==>输入、输出特征的梯度已经进行了归一化处理==>(LoadData.norm_data())
                       (2)==>path指定的.mat文件数据形状默认为(channel,height,width)
        """
        gradient_cache = {}
        current_layer = self.parsing_module(target_layer=target_layer)
        # 加载图像数据,图像数据的默认形状为(channel,height,width)
        image_data = sio.loadmat(image_message["path"])[image_message["key"]]
        image_data = torch.from_numpy(image_data).unsqueeze(0).to(self.Device)
        image_data.requires_grad = True
        target_label = (torch.ones((1, image_data.shape[2], image_data.shape[3]), dtype=torch.long) * target_value).to(self.Device)
        hook = current_layer.register_full_backward_hook(self.get_gradient_hook(cache=gradient_cache))
        self.Model.eval()
        output = self.Model(image_data)
        loss = self.Criterion(output, target_label)
        loss.backward()
        hook.remove()
        return gradient_cache['grad_input'], gradient_cache['grad_output']


class PerturbationSpectrum:
    """💞
    Description: 根据干净样本数据、真实标签数据以及模型权重生成对抗样本或对抗扰动,并对对抗扰动进行频谱分析(计算其径向频率曲线)
    """

    def __init__(self, receiver, clean_data, true_label, model_path, model_name, data_name, train_setting, epsilon: float = 0.5, except_label=255):
        """
        function     : 分析噪声的频谱属性
        receiver     : 类Receiver的实例
        clean_data   : 直接加载得到的图像数据,没有经过任何处理,数据形状为(H,W,C)
        true_label   : 直接加载得到的标签数据,没有经过任何处理,数据形状为(H,W),数值范围(O~N)其中N是类别个数
        model_path   : 完整的模型权重地址(XXX/XXX/model.pt)
        model_name   : 模型名称
        data_name    : 数据集名称
        train_setting: 训练设置
        epsilon      : 扰动幅度
        except_label : 忽略的标签
        """
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Model = torch.load(model_path, weights_only=False).to(self.Device)
        # 设置图像、标签数据
        self.Image = torch.from_numpy(LoadData.norm_data(clean_data).transpose(2, 0, 1)).unsqueeze(0).float().to(self.Device)
        true_label = true_label - 1
        true_label[true_label == -1] = except_label
        self.Label = torch.from_numpy(true_label).unsqueeze(0).to(self.Device)
        # 设置模型接收器(用来攻击模型)
        self.Receiver = receiver
        self.Receiver.reset(model=self.Model, except_label=except_label, settings=train_setting)
        # 设置类属性
        self.Epsilon = epsilon
        self.ModelName = model_name
        self.DataName = data_name

    def gen_adv_content(self, method, mode="noise"):
        """
        function: 根据攻击方法生成相应的对抗性噪声
        method  : 攻击方法
        mode    : 'noise'==>对抗性噪声、'adv'==>对抗样本
        return  : 对抗性噪声(channel,height,width)
        """
        print(f"ModelName==>{self.ModelName}")
        if method == "PGD":
            adv_images = self.Receiver.pgd_attack(image=self.Image, true_label=self.Label, root_dir="", epsilons=[self.Epsilon])
        elif method == "FGSM":
            adv_images = self.Receiver.target_fgsm_with_norm(image=self.Image, label=self.Label, root_dir="", epsilons=[self.Epsilon])
        else:
            adv_images = self.Receiver.cw_attack(image=self.Image, true_label=self.Label, root_dir="", epsilons=[self.Epsilon])
        adv_image = adv_images[f"{method}-{self.Epsilon}"]
        if mode == "noise":
            # 这样写是否合适????????
            return adv_image - self.Image.squeeze(0).cpu().numpy()
        elif mode == 'adv':
            return adv_image
        else:
            raise ValueError("模式写错辣!!!")

    @staticmethod
    def compute_radial_spectrum(image_data, remove_dc=True, use_power=True):
        """
        function  : 计算二维图像的径向平均频谱
        image_data: 图像数据(height,width)
        remove_dc : 是否去除直流分量
        use_power : 是否使用功率谱
        return    : frequency_radius  : 离散频率半径
                    spectral_intensity: 平均频率强度
        Note      : use_power:True==>使用功率谱|F|^2
                              False=>使用幅度谱|F|
        """
        if image_data.ndim != 2:
            raise ValueError(f"Image must be 2D, but got shape {image_data.shape}!!!")
        img = image_data.astype(np.float64).copy()
        if remove_dc:  # 去除直流分量
            img = img - np.mean(img)
        height, width = img.shape
        fft_map = np.fft.fft2(img)
        fft_map = np.fft.fftshift(fft_map)
        if use_power:
            spectrum_2d = np.abs(fft_map) ** 2
        else:
            spectrum_2d = np.abs(fft_map)
        center_y, cx = height // 2, width // 2
        y = np.arange(height) - center_y
        x = np.arange(width) - cx
        yy, xx = np.meshgrid(y, x, indexing='ij')
        # 计算每个频率点(u,v)到中心原点的欧几里得距离,对应公式
        rr = np.sqrt(xx ** 2 + yy ** 2)
        rr_int = np.round(rr).astype(np.int32)
        max_r = rr_int.max()
        # 把相同半径的所有像素的频谱能量spectrum_2d(u,v)累加起来
        radial_sum = np.bincount(rr_int.ravel(), weights=spectrum_2d.ravel(), minlength=max_r + 1)
        # 统计每个半径圈内有多少个像素点
        radial_count = np.bincount(rr_int.ravel(), minlength=max_r + 1)
        # 总能量除以总像素数,得到该半径下的平均能量S(r),使用np.maximum(..., 1)是为了防止某些半径没有像素时出现除以0的错误。
        spectral_intensity = radial_sum / np.maximum(radial_count, 1)
        frequency_radius = np.arange(len(spectral_intensity), dtype=np.int32)
        return frequency_radius, spectral_intensity

    @staticmethod
    def plot_radial_spectrum(dict_data, dataset, save_path, color, label, remove_dc=True, use_log=True, use_power=True, save_type='png'):
        """
        function : 接收字典数据,对每个模型的numpy数组在通道维度取均值后计算频谱,绘制曲线.
        dict_data: {'model1': np.array(C,H,W),...,'modelN': np.array(C,H,W)}
        dataset  : 数据集名称
        save_path: 图片保存路径(文件夹路径,如果没有则自动创建该路径)
        color    : 曲线的颜色
        label    : legend标签
        remove_dc: 是否去除直流分量(当设为True时,在计算频谱前会先减去图像的像素均值.这能消除频谱中心能量极高的零频率尖峰(直流分量)防止其撑爆纵坐标，从而让其他频率成分的细节能够清晰展现)
        use_log  : 是否使用对数坐标(当设为True时,会对计算出的频谱执行np.log1p,[即log(1+x)]变换.由于频谱能量跨度极大,取对数可以压缩数值范围,让高频段的微弱特征也能在图上清晰可见)
        use_power: 功率谱与幅度谱切换
        return   : None
        """
        if save_type not in ['png', 'pdf']:
            raise ValueError("保存类型写错辣!!!")
        set_matplotlib()
        plt.figure()
        for model_name, array in dict_data.items():
            if array.ndim != 3:
                raise ValueError(f"[Warning]:'{model_name}'的数组形状应为(C,H,W),实际为{array.shape}.")
            # (C,H,W)==>(H,W) 在通道维度取均值
            # print(f"{array.shape=}")
            mean_image = np.mean(array.astype(np.float64), axis=0)
            # print(f"{mean_image.shape=}")
            # 频谱横坐标、频谱强度
            spectral_coordinate, spectrum = PerturbationSpectrum.compute_radial_spectrum(image_data=mean_image, remove_dc=remove_dc, use_power=use_power)
            spectrum_to_plot = np.log1p(spectrum) if use_log else spectrum
            plt.plot(spectral_coordinate, spectrum_to_plot, linewidth=2, label=label[model_name], color=color[model_name])
        plt.xlabel("Frequency Radius", fontsize=24)
        plt.ylabel("Spectrum Magnitude", fontsize=24)
        plt.tick_params(axis='both', labelsize=24)  # 调整刻度字体大小
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, linestyle='--')
        axis = plt.gca()
        # 设置坐标轴刻度间隔
        axis.xaxis.set_major_locator(MultipleLocator(50))
        axis.yaxis.set_major_locator(MultipleLocator(3))
        plt.tight_layout()
        # 如果没有该路径自动创建该路径
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_path is not None:
            if save_type == 'png':
                plt.savefig(fr"{save_path}/{dataset}.png", dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='#FFFFFF')
            else:
                plt.savefig(fr"{save_path}/{dataset}.pdf", format="pdf", pad_inches=0.05, facecolor='#FFFFFF')
        plt.show()


def parsing_json_files(data_name, model_name, epsilon) -> None:
    """💞
    function  : 解析模型在干净样本以及对抗样本上的测试结果
    data_name : 数据集名称
    model_name: 模型名称
    epsilon   : 扰动幅度
    return    : None
    """
    script_path = Path(__file__).resolve().parent.parent
    json_path = fr"{script_path}/Result/Json/{model_name}-{data_name}.json"
    with open(json_path, "r", encoding="utf-8") as file:
        json_str = file.read()
    attack_result = json.loads(json_str)
    # 显示干净样本整体测试结果(epsilon=0.00)
    clean_description = f" 模型{model_name}在数据集{data_name}上测试(干净样本),每个类别以及统计量的均值、标准差如下:"
    receiver = Receiver(model_name=f"{model_name}", save_path=rf"{script_path}/Result", data_name=data_name)
    clean_result = receiver.get_epsilon_result(dict_data=attack_result, epsilon=0)
    receiver.record_result_to_log(description=clean_description, result=clean_result, mode='summary', is_print=True)
    # 显示对抗样本整体测试结果(epsilon=0.0X)
    attack_description = f" 模型{model_name}在数据集{data_name}上攻击(epsilon={epsilon}),每个类别以及统计量的均值、标准差如下:"
    fgsm_attack_result = receiver.get_epsilon_result(dict_data=attack_result, epsilon=epsilon)  # noqa
    receiver.record_result_to_log(description=attack_description, result=fgsm_attack_result, mode='summary', is_print=True)
    # 不同攻击方法结果对比
    fgsm = receiver.get_epsilon_result(dict_data=attack_result, method='FGSM', epsilon=epsilon)
    pgd = receiver.get_epsilon_result(dict_data=attack_result, method='PGD', epsilon=epsilon)
    cw = receiver.get_epsilon_result(dict_data=attack_result, method='CW', epsilon=epsilon)
    methods_result = {f'FGSM-{epsilon}': fgsm, f'PGD-{epsilon}': pgd, f'CW-{epsilon}': cw}
    receiver.record_result_to_log(description='', result=methods_result, mode='contrast', is_print=True)


def set_matplotlib() -> None:
    """💞
    function: 设置Matplotlib设置(加载配置、加载设置字体文件)
    return  : None
    """
    script_path = os.path.dirname(__file__)  # 当前Python脚本的完整路径
    if os.name != "nt" and os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            matplotlib.use("Agg")
    font_manager.fontManager.addfont(script_path + '/SongTimes.ttf')  # 将字体添加到搜索路径中
    # 将字体添加到搜索路径中
    font_path = script_path + '/SongTimes.ttf'
    font_manager.fontManager.addfont(font_path)
    # 设置全局默认字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Song Times'
    # plt.rcParams.update({"mathtext.fontset": "stix", "mathtext.bf": "bold", "axes.labelweight": "bold", "font.weight": "bold", })  # noqa
    plt.rcParams.update({"mathtext.fontset": "stix", "mathtext.default": "rm"})  # noqa


def requirement(file_path: str = './requirement.txt') -> None:
    """💞
    function : 将当前环境信息及包信息写在文档file_path中
    file_path: 文档地址
    return   : None
    """
    with open(file_path, 'w') as file:
        file.write(f"------------Environmental Information------------\n")
        file.write(f"Python  Version:   {sys.version}\n")
        file.write(f"PyTorch Version:   {torch.__version__}\n")
        file.write(f"CUDA    Version:   {torch.version.cuda}\n")
        file.write(f"------------Package Information------------------\n")
        output = subprocess.check_output(['pip', 'list'])
        clean_output_str = '\n'.join(line for line in output.decode('utf-8').splitlines() if line)
        file.write(clean_output_str)


def show_env_message() -> None:
    """💞
    function: 显示程序所在环境信息
    return  : None
    """
    red = "\033[91m"
    blue = "\033[94m"
    green = "\033[92m"
    reset = "\033[0m"
    print("=" * 120)
    print(f"{red}Environmental information{reset}")
    print(f"{green}Python  Version{reset}: {blue}{sys.version}{reset}")
    print(f"{green}PyTorch Version{reset}: {blue}{torch.__version__}{reset}")
    print(f"{green}CUDA  Available{reset}: {blue}{torch.cuda.is_available()}")
    print(f"{green}CUDA    Version{reset}: {blue}{torch.version.cuda}{reset}")
    # GPU Information
    if torch.cuda.is_available():
        print(f"{red}GPU Information{reset}")
        for i in range(torch.cuda.device_count()):
            print(f"{green}Device {i}{reset}")
            print(f"{green}Device Name{reset}      : {blue}{torch.cuda.get_device_name(i)}{reset}")
            print(
                f"{green}Device Memory{reset}    : {blue}"
                f"{torch.cuda.get_device_properties(i).total_memory / (1024 * 1024 * 1024):.2f} GB{reset}")
    else:
        print(f"{red}No GPU detected.{reset}")


def draw_ground_truth(true_label: numpy.ndarray, predict_label: np.ndarray, dataset_name: str, is_show: bool = False,
                      save_path: str | None = None, file_name: str | None = None, title: str | None = None) -> None:
    """💞
    function     : 根据真实标签以及预测标签绘制假彩色图像
    true_label   : (1,height,width)==>真实标签,数值范围为(0,1,2,3,...,num_class),其中0就是背景标签
    predict_label: (1,height,width)==>预测标签,数值范围为(0,1,2,3,...,num_class-1)
    dataset_name : 数据集名称
    is_show      : 是否显示matplotlib界面
    save_path    : 图片保存地址
    file_name    : 图像保存名称
    title        : Figure界面的标题
    return       : None
    """

    def hex_to_rgb(hex_color):
        """👌
        function : 将十六进制颜色转换为RGB
        hex_color: #??????
        return   :
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[index:index + 2], 16) / 255.0 for index in (0, 2, 4))

    if dataset_name not in ["Salinas", "PaviaU", "Indian_pines", "Indian_pines_corrected", "Salinas_corrected"]:
        raise ValueError(f"尚不支持的数据集,请为数据集{dataset_name}添加颜色映射!!!")
    colors = {"Salinas": ["#6F3B22", "#0000C1", "#CD5700", "#00DE5E", "#844176", "#5196C2", "#5FDD83", "#304F55",
                          "#CDDE01", "#CDDE60", "#CC00C1", "#5000C2", "#0095C1", "#00DE01", "#8A983D", "#51A82E"],
              "PaviaU": ["#D8BFD8", "#00FF00", "#00FFFF", "#2D8A56", "#FF00FF", "#FFA500", "#9F1FEF", "#FF0000", "#FFFF00"],  # noqa
              "Indian_pines": ["#DB5E56", "#DB9056", "#DBC256", "#C2DB56", "#91DB56", "#5FDB56", "#56DB7F", "#56DBB1",
                               "#56D3DB", "#56A1DB", "#566FDB", "#6F56DB", "#A056DB", "#D256DB", "#DB56B2", "#DB5680"],
              "Indian_pines_corrected": ["#DB5E56", "#DB9056", "#DBC256", "#C2DB56", "#91DB56", "#5FDB56", "#56DB7F", "#56DBB1",
                                         "#56D3DB", "#56A1DB", "#566FDB", "#6F56DB", "#A056DB", "#D256DB", "#DB56B2", "#DB5680"],
              "Salinas_corrected": ["#6F3B22", "#0000C1", "#CD5700", "#00DE5E", "#844176", "#5196C2", "#5FDD83", "#304F55",
                                    "#CDDE01", "#CDDE60", "#CC00C1", "#5000C2", "#0095C1", "#00DE01", "#8A983D", "#51A82E"]}
    true_label = true_label.squeeze(0)
    predict_label = predict_label.squeeze(0)
    color = colors[dataset_name]
    height, width = true_label.shape
    rgb_color = [hex_to_rgb(c) for c in color]
    # 初始化RGB图像(height,width,3)
    pred_rgb = np.zeros((height, width, 3))
    background_mask = (true_label == 0)  # 背景像素掩码
    for i in range(len(color)):
        mask = (predict_label == i) & (~background_mask)  # 只在非背景位置绘制预测结果
        pred_rgb[mask] = rgb_color[i]
    # 绘制结果
    set_matplotlib()
    plt.figure()
    plt.imshow(pred_rgb, interpolation='none')
    plt.axis('off')
    # 保存图像
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = dataset_name if file_name is None else file_name
        plt.savefig(fr"{save_path}/{file_name}-GroundTruth.png", dpi=300, bbox_inches='tight', pad_inches=0)
    if is_show:
        plt.title(f'数据{dataset_name}预测结果的假彩色图像' if title is None else title, fontsize=16)
        plt.show()


def plot_model_robustness_curves(root_path: str, method: str, model_name: dict, dataset_name: str, epsilons: List[float], save_path: str, indicators='OA', save_type="png"):
    """💞
    function    : 绘制不同模型在不同攻击扰动强度下的整体精度变化曲线
    root_path   : 对抗样本测试结果的根目录地址
                  root_path
                      |-model-data.json
                      | ......
                      |-model-data.json
    method      : 攻击方式==>['CW','FGSM','PGD']
    model_name  : 模型名称及其相应的图例名称(legend)==>{'model_name': legend_name, ..., 'model_name': legend_name}
    dataset_name: 数据集名称
    epsilons    : 扰动幅度
    save_path   : 图像保存地址,只是一个目录地址,不包含文件名称(XXX/XXX/XXX)
    indicators  : 选中的指标
    save_type   : 图像保存类型==>['png','pdf']
    return      : None
    """
    set_matplotlib()
    if method not in ['FGSM', 'CW', 'PGD']:
        raise ValueError("攻击方式写错辣!!!")
    if indicators not in ['OA', 'AA', 'Kappa']:
        raise ValueError("指标写错辣!!!")
    if save_type not in ['png', 'pdf']:
        raise ValueError("保存类型写错辣!!!")
    if len(model_name) > 10:
        raise ValueError("最多只能绘制十个模型的体精度变化曲线!!!,若要增加上限,为color添加更多的颜色值!!!")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    color = ["black", "blue", "#4361ee", "#ff0000", "#54d691", "#008000", "#9800ff", "#b4159d", "#ff66ff", "#ff9700"]
    index = {"OA": -3, "AA": -2, "Kappa": -1}[indicators]
    y_title = {"OA": "Overall Accuracy(%)", "AA": "Average Accuracy(%)", "Kappa": "Kappa(%)"}[indicators]
    # 创建界面
    figure, axis = plt.subplots(figsize=(6, 6))
    # 解析数据
    for i, model in enumerate(model_name.keys()):
        json_path = root_path + fr"/{model}-{dataset_name}.json"
        with open(json_path, "r", encoding="utf-8") as file:
            json_str = file.read()
        attack_result = json.loads(json_str)
        values = []
        for epsilon in epsilons:
            means, _ = Receiver.calculate_mean_std(Receiver.get_epsilon_result(dict_data=attack_result, method=method, epsilon=epsilon))
            values.append(means[index])
        axis.plot(epsilons, values, marker='s', linewidth=2, markersize=6, label=model_name[model], color=color[i])
    axis.set_aspect('auto')
    axis.set_ylim(0, 102)
    axis.set_yticks(np.arange(10, 102, 10))
    x_ticks = np.arange(0, 0.51, 0.1)
    axis.set_xticks(x_ticks)
    axis.set_xticklabels([f"{x:.1f}" if x != 0 else "0" for x in x_ticks])
    axis.tick_params(axis='both', labelsize=28)
    axis.set_xlabel('Attack Epsilon', fontsize=28)
    axis.set_ylabel(y_title, fontsize=28)
    axis.legend(loc='lower left', frameon=False)
    axis.grid(False)
    plt.tight_layout()
    file_name = {"OA": "OverallAccuracy", "AA": "AverageAccuracy", "Kappa": "Kappa"}[indicators]
    if save_type == "png":
        plt.savefig(fr"{save_path}/{file_name}-{dataset_name}.png", dpi=300, facecolor='#FFFFFF')
        print(fr"图片已保存至==>{save_path}/{file_name}-{dataset_name}.png")
    if save_type == "pdf":
        plt.savefig(fr"{save_path}/{file_name}-{dataset_name}.pdf", format="pdf", pad_inches=0.02, facecolor='#FFFFFF')
        print(fr"图片已保存至==>{save_path}/{file_name}-{dataset_name}.pdf")
    plt.show()


def show_mean_var(lines_data: np.ndarray, line_labels: list, region: List[float] | None, location: Tuple[float, ...] | None, connection_position: Tuple[int, ...] | None,
                  save_path: str, file_name, is_ax_ins=True, is_legend=False) -> None:
    """💞
    function           : 绘制多条折线,并在指定位置绘制局部放大图
    lines_data         : 折线数据
    line_labels        : 每条折线数据标签名称
    region             : 局部放大的位置
    location           : 局部放大图的位置
    connection_position: 连接线的位置
    save_path          : 图像数据的保存地址(没有该地址则创建该地址)
    file_name          : 保存的图片文件名称(XXX.PNG)
    is_ax_ins          :
    legend_label       : 显示图例标签
    return             : None
    """
    set_matplotlib()  # 如果有自定义设置请取消注释
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.set_facecolor('#FFFFFF')
    x = np.arange(0, lines_data.shape[1])
    colors = plt.cm.turbo(np.linspace(0.2, 0.8, len(line_labels)))  # noqa
    numeric_items = [(float(label), colors[i]) for i, label in enumerate(line_labels) if not isinstance(label, str)]
    numeric_items = sorted(numeric_items, key=lambda item: item[0])
    numeric_labels = [item[0] for item in numeric_items]
    numeric_colors = [item[1] for item in numeric_items]
    epsilon_colormap = LinearSegmentedColormap.from_list("epsilon_cmap", numeric_colors, N=256)
    norm = Normalize(vmin=min(numeric_labels), vmax=max(numeric_labels))

    if is_ax_ins:
        ax_ins = inset_axes(axis, width=f"{location[2] * 100}%", height=f"{location[3] * 100}%", bbox_to_anchor=(location[0], location[1], 1, 1),
                            bbox_transform=axis.transAxes, loc='lower left')
    # 绘制主体折线和放大折线
    for i in range(lines_data.shape[0]):
        line_label = line_labels[i]
        label = None if type(line_label) is not str else line_label

        color = colors[i] if type(line_label) is not str else "black"
        line_style = '-' if type(line_label) is not str else "--"
        axis.plot(x, lines_data[i, :], color=color, label=label, alpha=0.8, linewidth=1.5, linestyle=line_style)
        if is_ax_ins:
            ax_ins.plot(x, lines_data[i, :], color=color, alpha=0.8, linewidth=1.5, linestyle=line_style)  # noqa
    if is_ax_ins:
        ax_ins.set_xlim(region[0], region[1])
        ax_ins.set_ylim(region[2], region[3])
        mark_inset(axis, ax_ins, loc1=connection_position[0], loc2=connection_position[1], fc="none", ec="0.5", linestyle="--")
    if is_legend:
        axis.legend(fontsize=18)
    # 样式调整
    axis.tick_params(axis='both', labelsize=28)
    axis.set_xlabel('Spectral Band', fontsize=28)
    axis.grid(True, linestyle=':', alpha=0.5)
    # 先调整主图四周留白,确保右侧留出足够空间给颜色条
    figure.subplots_adjust(left=0.11, right=0.9, bottom=0.12, top=0.98)
    # --- 核心修改部分：使用 make_axes_locatable 动态生成和主图等高的 cax ---
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    sm = ScalarMappable(norm=norm, cmap=epsilon_colormap)
    sm.set_array([])
    color_bar = figure.colorbar(sm, cax=cax)
    color_bar.set_label(r"Perturbation magnitude", fontsize=20)  # noqa
    color_bar.set_ticks(numeric_labels)
    color_bar.ax.tick_params(labelsize=16)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_full_path = os.path.join(save_path, file_name)
    if file_name[-3:] == "pdf":
        plt.savefig(file_full_path, format="pdf", pad_inches=0.02, facecolor='#FFFFFF')
    elif file_name[-3:] == "png":
        plt.savefig(file_full_path, format="png", dpi=300, pad_inches=0.02, facecolor='#FFFFFF')
    plt.show()


def draw_ablation_line(clean_data, adv_data, data_name, module_name, save_path, save_type='png') -> None:
    """💞
    function   :
    clean_data : 干净数据
    adv_data   : 对抗样本数据
    data_name  : 数据集名称
    module_name: 模块名称
    save_path  : 保存路径(没有该地址则创建该地址)
    save_type  : 数据保存类型('png'、'pdf')
    return     : None
    """
    if save_type not in ['png', 'pdf']:
        raise ValueError("保存类型写错辣!!!")
    set_matplotlib()
    x = np.arange(len(clean_data))
    fig, axes = plt.subplots()
    axes.plot(x, adv_data, color="red", linewidth=1.4, label="Adversarial Spectral")
    axes.plot(x, clean_data, color="blue", linewidth=1.4, label="Original Spectral")
    axes.set_xlim(-len(x) * 0.01, len(x) * 1.01)
    y_min = min(clean_data.min(), adv_data.min())
    y_max = max(clean_data.max(), adv_data.max())
    margin = (y_max - y_min) * 0.1
    axes.set_ylim(y_min - margin, y_max + margin)
    axes.grid(True, linewidth=0.8, linestyle="--", alpha=0.9)
    axes.set_axisbelow(True)
    axes.set_aspect('auto')
    axes.xaxis.set_major_locator(MultipleLocator(40))
    axes.yaxis.set_major_locator(MultipleLocator(0.1))
    axes.set_xlabel("Spectral Bands", fontsize=28, labelpad=4)
    axes.tick_params(labelsize=28)
    axes.set_ylabel("Reflectance", fontsize=28, labelpad=4)
    location = "upper left" if data_name == "PaviaU" else "upper right"
    axes.legend(fontsize=14, loc=location, edgecolor="#C9D6E3")
    plt.tight_layout()
    # 若没有该地址则创建该地址
    save_path = save_path + fr"/{data_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_type == 'png':
        plt.savefig(fr"{save_path}/{module_name}.png", dpi=300, bbox_inches='tight', pad_inches=0.08, facecolor='#FFFFFF')
    else:
        plt.savefig(fr"{save_path}/{module_name}.pdf", format="pdf", pad_inches=0.08, facecolor='#FFFFFF')
    plt.show()


if __name__ == '__main__':
    print("5211314")
