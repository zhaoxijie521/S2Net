import torch.nn.functional as func
import math
from einops import rearrange
import torch
import torch.nn as nn

BASE = "Baseline"
M1 = "SGS"
M2 = "SAC"
M3 = "FDGP"


class SpectralAdaptiveCorrectionModule(nn.Module):
    """🚀
    Description: Spectral Adaptive Correction Module
    """

    def __init__(self, num_features, ablation, eps=1e-5):
        super().__init__()
        self.eps = eps
        hidden = 256
        self.ablation = ablation
        self.channel_weight = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=3), nn.Conv2d(num_features, hidden, kernel_size=1, bias=False),
                                            nn.MaxPool2d(kernel_size=3, stride=3),
                                            nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, num_features, kernel_size=1, bias=False), nn.Sigmoid())
        self.channel_bias = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=3),
                                          nn.Conv2d(num_features, hidden, kernel_size=1, bias=False),
                                          nn.MaxPool2d(kernel_size=3, stride=3),
                                          nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, num_features, kernel_size=1, bias=False), nn.GELU())
        self.BatchNorm = nn.BatchNorm2d(num_features)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        function  : Forward
        input_data: (batch,channel,height,width)
        return    : (batch,channel,height,width)
        """
        if self.ablation in [f"{BASE}", f"{BASE}+{M1}", f"{BASE}+{M3}", f"{BASE}+{M1}+{M3}"]:  # noqa
            return self.BatchNorm(input_data)
        elif self.ablation in [f"{BASE}+{M2}", f"{BASE}+{M3}+{M2}", f"{BASE}+{M1}+{M2}", f"{BASE}+{M1}+{M3}+{M2}"]:  # noqa
            mean = input_data.mean(dim=[0, 2, 3], keepdim=True)
            var = input_data.var(dim=[0, 2, 3], keepdim=True, unbiased=False)
            norm_data = (input_data - mean) / torch.sqrt(var + self.eps)
            weight = self.channel_weight(norm_data)
            bias = self.channel_bias(norm_data)
            result = (1 + weight) * norm_data + bias
            return result
        else:
            raise ValueError("消融实验模式写错辣!!!")


class FrequencyDomainGlobalPerceptionModule(nn.Module):
    """
    Description: Frequency Domain Global Perception Module(频域全局感知模块)
    """

    def __init__(self, in_channel, height, width, kernel_size, is_residual, ablation, filter_mode='adaptive', bias=True, hidden=256, cutoff=0.5):
        """
        function   : Initialize
        in_channel : 输入数据通道数
        height     : 输入图像高度
        width      : 输入图像宽度
        kernel_size: 卷积核大小
        is_residual: 是否设置残差连接
        filter_mode: 滤波模式:'low'==>低通滤波、'high'==>高通滤波、'adaptive'==>自适应滤波
        bias       : 是否设置偏置
        hidden     : 隐藏层维度
        cutoff     : 截止频率
        """
        super().__init__()
        # 降噪模块
        self.ACD1 = SpectralAdaptiveCorrectionModule(in_channel, ablation=ablation)
        self.ACD2 = SpectralAdaptiveCorrectionModule(in_channel, ablation=ablation)
        # 通道缩放参数
        self.LayerScale1 = nn.Parameter(0.02 * torch.ones(in_channel), requires_grad=True)
        self.LayerScale2 = nn.Parameter(0.02 * torch.ones(in_channel), requires_grad=True)
        # 残差连接权重
        self.Alpha1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Alpha2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.IsResidual = is_residual
        # 频域滤波设置
        self.Height = height
        self.Width = width
        self.FilterMode = filter_mode
        self.cutoff = cutoff
        if filter_mode not in ['low', 'high', 'adaptive']:
            raise ValueError('模式写错辣!!!')
        if filter_mode in ['low', 'high']:
            self.ImagConv = nn.Identity()
            self.RealConv = nn.Identity()
        else:
            self.RealConv = nn.Sequential(nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, padding=1),
                                          nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=3, padding=1, groups=in_channel * 2),
                                          nn.GELU(),
                                          nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=1))
            self.ImagConv = nn.Sequential(nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, padding=1),
                                          nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=3, padding=1, groups=in_channel * 2),
                                          nn.GELU(),
                                          nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=1))
        self.complex_weight = nn.Parameter(torch.rand(1, height, width // 2 + 1, 2, dtype=torch.float32), requires_grad=True)
        self.register_buffer("mask", self.build_mask())
        # "前馈神经网络"
        self.FFN = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=hidden, kernel_size=kernel_size[0], stride=1, padding=kernel_size[0] // 2, bias=bias, groups=8),
                                 nn.Conv2d(hidden, hidden, kernel_size[1], 1, padding=kernel_size[1] // 2, bias=True, groups=hidden),
                                 nn.Conv2d(in_channels=hidden, out_channels=in_channel, kernel_size=kernel_size[2], stride=1, padding=kernel_size[2] // 2, bias=bias, groups=8),
                                 nn.GELU())

    def build_mask(self):
        """
        function: 根据不同的模式生成频率掩码
        return  : 频率掩码(1,1,height,width)
        """
        height, width = self.Height, self.Width // 2 + 1
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        center_y = height // 2
        center_x = 0
        r = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        r = r / r.max()
        if self.FilterMode == 'low':
            mask = (r <= self.cutoff).float()
        elif self.FilterMode == 'high':
            mask = (r >= self.cutoff).float()
        else:
            mask = torch.ones_like(r)
        return mask.unsqueeze(0).unsqueeze(0)

    def spectre_former(self, input_data):
        """
        function  : 频域滤波
        input_data: (batch,embed,height,width)
        return    : (batch,embed,height,width)
        """
        input_data = torch.fft.rfft2(input_data, dim=(-2, -1), norm='ortho')
        real_data = self.RealConv(input_data.real)
        imag_data = self.ImagConv(input_data.imag)
        input_data = torch.complex(real_data, imag_data)
        weight = torch.view_as_complex(self.complex_weight)
        if self.FilterMode in ['low', 'high']:
            input_data = input_data * self.mask
        elif self.FilterMode == 'adaptive':
            input_data = input_data * weight
        else:
            raise ValueError("模式写错辣!!!")
        input_data = torch.fft.irfft2(input_data, s=(self.Height, self.Width), dim=(-2, -1), norm='ortho')
        return input_data

    def forward(self, data):
        """😄
        function: forward
        data    : (batch,embed,height,width)
        return  : (batch,embed,height,width)
        """
        if self.IsResidual:
            data = data * self.Alpha1 + self.spectre_former(self.ACD1(data)) * self.LayerScale1.unsqueeze(-1).unsqueeze(-1)
            data = data * self.Alpha2 + self.FFN(self.ACD2(data)) * self.LayerScale2.unsqueeze(-1).unsqueeze(-1)
        else:
            data = self.spectre_former(self.ACD1(data)) * self.LayerScale1.unsqueeze(-1).unsqueeze(-1)
            data = self.FFN(self.ACD2(data)) * self.LayerScale2.unsqueeze(-1).unsqueeze(-1)
        return data


class PyramidModule(nn.Module):
    """👍
    Description: 金字塔层模块
    """

    def __init__(self, in_channel, height, width, ablation, hidden=256, filter_mode='adaptive', embed=64, cutoff=0.5):
        """😄
        function  : Initialize
        in_channel: 输入通道数
        height    : 图像高度
        width     : 图像宽度
        embed     : 嵌入维度
        block_num : Spectre Block块个个数
        """
        super().__init__()
        # 可变形卷积块
        if ablation in [f"{BASE}+{M1}", f"{BASE}+{M1}+{M3}", f"{BASE}+{M1}+{M2}", f"{BASE}+{M1}+{M3}+{M2}"]:  # noqa
            self.PreModule1 = nn.Sequential(SpatialGradientSmoothingModule(in_channel, embed, 25, stride=2), SpectralAdaptiveCorrectionModule(embed, ablation=ablation), nn.GELU())
            self.PreModule2 = nn.Sequential(SpatialGradientSmoothingModule(embed, embed, 25, stride=2), SpectralAdaptiveCorrectionModule(embed, ablation=ablation), nn.GELU())
            self.PreModule3 = nn.Sequential(SpatialGradientSmoothingModule(embed, embed, 25, stride=2), SpectralAdaptiveCorrectionModule(embed, ablation=ablation), nn.GELU())
        elif ablation in [f"{BASE}", f"{BASE}+{M2}", f"{BASE}+{M3}", f"{BASE}+{M3}+{M2}"]:  # noqa
            self.PreModule1 = nn.Sequential(nn.Conv2d(in_channel, embed, 5, stride=2, padding=2), SpectralAdaptiveCorrectionModule(embed, ablation=ablation), nn.GELU())
            self.PreModule2 = nn.Sequential(nn.Conv2d(embed, embed, 5, stride=2, padding=2), SpectralAdaptiveCorrectionModule(embed, ablation=ablation), nn.GELU())
            self.PreModule3 = nn.Sequential(nn.Conv2d(embed, embed, 5, stride=2, padding=2), SpectralAdaptiveCorrectionModule(embed, ablation=ablation), nn.GELU())
        else:
            raise ValueError("模式写错辣!!!")
        # SpectreBlock参数
        height_list, width_list = [height], [width]
        for _ in range(3):
            temp_height = height_list[-1] // 2 if height_list[-1] % 2 == 0 else height_list[-1] // 2 + 1
            temp_width = width_list[-1] // 2 if width_list[-1] % 2 == 0 else width_list[-1] // 2 + 1
            height_list.append(temp_height)
            width_list.append(temp_width)
        for index in range(0, 4):
            setattr(self, f"height{index}", height_list[index])
            setattr(self, f"width{index}", width_list[index])
        # Spectre Block
        # Spectre Block
        spectre_param1 = dict(in_channel=embed, height=self.height1, width=self.width1, kernel_size=(3, 3, 3), is_residual=True,
                              filter_mode=filter_mode, bias=True, hidden=hidden, cutoff=cutoff, ablation=ablation)
        self.SpectreBlock1 = nn.Sequential(FrequencyDomainGlobalPerceptionModule(**spectre_param1))
        spectre_param2 = dict(in_channel=embed, height=self.height2, width=self.width2, kernel_size=(5, 5, 5), is_residual=False,
                              filter_mode=filter_mode, bias=True, hidden=hidden, cutoff=cutoff, ablation=ablation)
        self.SpectreBlock2 = nn.Sequential(FrequencyDomainGlobalPerceptionModule(**spectre_param2))
        spectre_param3 = dict(in_channel=embed, height=self.height3, width=self.width3, kernel_size=(7, 7, 7), is_residual=False,
                              filter_mode=filter_mode, bias=True, hidden=hidden, cutoff=cutoff, ablation=ablation)
        self.SpectreBlock3 = nn.Sequential(FrequencyDomainGlobalPerceptionModule(**spectre_param3))
        # 卷积块
        self.Conv1 = nn.Sequential(nn.Conv2d(embed, embed, kernel_size=7, padding=3), SpectralAdaptiveCorrectionModule(embed, ablation=ablation), nn.ReLU())
        self.Conv2 = nn.Sequential(nn.Conv2d(embed, embed, kernel_size=5, padding=2), SpectralAdaptiveCorrectionModule(embed, ablation=ablation), nn.ReLU())
        self.Conv3 = nn.Sequential(nn.Conv2d(embed, in_channel, kernel_size=3, padding=1), SpectralAdaptiveCorrectionModule(in_channel, ablation=ablation), nn.GELU())
        if ablation in [f"{BASE}", f"{BASE}+{M1}", f"{BASE}+{M2}", f"{BASE}+{M1}+{M2}"]:  # noqa
            self.SpectreBlock3 = nn.Identity()
            self.SpectreBlock2 = nn.Identity()
            self.SpectreBlock1 = nn.Identity()

        # 系数
        self.Alpha1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Alpha2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Alpha3 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Gama1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Gama2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.Gama3 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, input_data):
        """😄
        function  : Forward
        input_data: (batch,channel,height,width)
        return    : (batch,channel,height,width)
        """
        # 利用可变形卷积进行下采样
        pre_data1 = self.PreModule1(input_data)  # (1,C,H//2,W//2)
        pre_data2 = self.PreModule2(pre_data1)  # (1,C,H//4,W//4)
        pre_data3 = self.PreModule3(pre_data2)  # (1,C,H//8,W//8)
        # 快速傅里叶变换
        spectre1 = self.SpectreBlock1(pre_data1)  # (1,C,H//2,W//2)
        spectre2 = self.SpectreBlock2(pre_data2)  # (1,C,H//4,W//4)
        spectre3 = self.SpectreBlock3(pre_data3)  # (1,C,H//8,W//8)
        # 残差连结
        mode = 'bilinear'
        # 第一级
        up_sample1 = func.interpolate(self.Gama1 * pre_data3 + self.Alpha1 * spectre3, size=(self.height2, self.width2), mode=mode, align_corners=False)
        residual1 = self.Conv1(up_sample1)
        # 第二级
        up_sample2 = func.interpolate(self.Gama2 * residual1 + self.Alpha2 * spectre2, size=(self.height1, self.width1), mode=mode, align_corners=False)
        residual2 = self.Conv2(up_sample2)
        # 第三级
        up_sample3 = func.interpolate(self.Gama3 * residual2 + self.Alpha3 * spectre1, size=(self.height0, self.width0), mode=mode, align_corners=False)
        residual3 = self.Conv3(up_sample3)
        return residual3


class SpatialGradientSmoothingModule(nn.Module):
    """👍Spatial Gradient Smoothing Module
    Description: 结合可变形卷积(Deformable Convolution)的思想,通过预测采样点的偏移量来动态调整卷积核的采样位置,实现梯度平滑下采样
    """

    def __init__(self, in_channel, out_channel, sample_number, stride=1, bias=None):
        """😄
        function     : Initialize
        in_channel   : 输入通道数
        out_channel  : 输出通道数
        sample_number: 采样点数
        stride       : 步长
        bias         : 卷积层是否使用偏置项
        """
        super(SpatialGradientSmoothingModule, self).__init__()
        self.stride = stride
        self.OffsetConv = nn.Conv2d(in_channel, 2 * sample_number, kernel_size=3, padding=1, stride=stride)  # 预测每个采样点的偏移量
        # 初始化权重为0,使得在训练的初始阶段,预测层不会对输入特征图产生过大的影响
        nn.init.constant_(self.OffsetConv.weight, 0)
        # 为OffsetConv层注册一个后向传播钩子函数,self.adjusting_gradient():指定的钩子函数,用于在后向传播过程中调整梯度的大小.
        # 将一个张量注册为模型的缓冲区(buffer)这行代码的主要功能是将generate_initial_offset()方法生成的张量initial_offset注册为模型的一部分,
        # 缓冲区不会参与梯度更新.这样就不用每次前向传播都需要计算了
        self.register_buffer("initial_offset", self.generate_initial_offset(sample_number=sample_number))
        self.OutConv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(sample_number, 1), stride=(sample_number, 1), bias=bias))

    def forward(self, input_data):
        """😄
        function  : Forward(可变形卷积层的前向传播过程)
        input_data: (batch,channel,height,width)
        return    : (batch,channel,height',width')
        """
        offset = self.OffsetConv(input_data)  # (batch,2*sample_number,height',width')==>生成每个采样点的偏移量
        dtype = offset.data.type()
        sample_number = offset.size(1) // 2  # 采样点个数
        sample_position = self.get_position(offset, dtype)  # (batch,2*sample_number,height',width')
        sample_position = sample_position.contiguous().permute(0, 2, 3, 1)  # (batch,height',width',2*sample_number)
        # detach():将张量从计算图中分离出来,返回一个新的张量,该张量与原始张量共享数据,但不会参与梯度计算
        left_up = sample_position.detach().floor()  # 向下取整(batch,height',width',2*sample_number)
        right_down = left_up + 1
        # 使用torch.clamp()将位置限制在输入特征图的有效范围内(高度、宽度的边界内),下面的数据形状都是(batch,height',width',2*sample_number)
        left_up = torch.cat([torch.clamp(left_up[..., :sample_number], 0, input_data.size(2) - 1),
                             torch.clamp(left_up[..., sample_number:], 0, input_data.size(3) - 1)], dim=-1).long()
        right_down = torch.cat([torch.clamp(right_down[..., :sample_number], 0, input_data.size(2) - 1),
                                torch.clamp(right_down[..., sample_number:], 0, input_data.size(3) - 1)], dim=-1).long()
        left_down = torch.cat([left_up[..., :sample_number], right_down[..., sample_number:]], dim=-1)
        right_up = torch.cat([right_down[..., :sample_number], left_up[..., sample_number:]], dim=-1)
        sample_position = torch.cat([torch.clamp(sample_position[..., :sample_number], 0, input_data.size(2) - 1),
                                     torch.clamp(sample_position[..., sample_number:], 0, input_data.size(3) - 1)], dim=-1)
        # 计算双线性插值中使用的权重(weight_left_top、weight_right_bottom、weight_left_bottom、weight_right_top),权重用于确定每个采样点从其周围的四个最近整数坐标点
        # (左上left_top、右下right_bottom、左下left_bottom、右上right_top)获取特征的贡献度,允许对非整数位置的采样点进行平滑的特征提取.
        # 权重的形状均为:(batch,height',width',sample_number)
        weight_left_up = ((1 + (left_up[..., :sample_number].type_as(sample_position) - sample_position[..., :sample_number])) *
                          (1 + (left_up[..., sample_number:].type_as(sample_position) - sample_position[..., sample_number:])))
        weight_right_down = ((1 - (right_down[..., :sample_number].type_as(sample_position) - sample_position[..., :sample_number])) *
                             (1 - (right_down[..., sample_number:].type_as(sample_position) - sample_position[..., sample_number:])))
        weight_left_down = ((1 + (left_down[..., :sample_number].type_as(sample_position) - sample_position[..., :sample_number])) *
                            (1 - (left_down[..., sample_number:].type_as(sample_position) - sample_position[..., sample_number:])))
        weight_right_up = ((1 - (right_up[..., :sample_number].type_as(sample_position) - sample_position[..., :sample_number])) *
                           (1 + (right_up[..., sample_number:].type_as(sample_position) - sample_position[..., sample_number:])))
        # 根据给定的采样点位置从输入特征图中提取特征,形状均为(batch,channel,height',width',sample_number)
        # feature[0,:,:,:,i]===>对于第i个采样点在特征图每个位置上获得的特征
        feature_left_up = self.extract_feature(input_data, left_up, sample_number)
        feature_right_down = self.extract_feature(input_data, right_down, sample_number)
        feature_left_down = self.extract_feature(input_data, left_down, sample_number)
        feature_right_up = self.extract_feature(input_data, right_up, sample_number)
        # 最终的特征
        final_feature = (weight_left_up.unsqueeze(dim=1) * feature_left_up + weight_right_down.unsqueeze(dim=1) * feature_right_down +
                         weight_left_down.unsqueeze(dim=1) * feature_left_down + weight_right_up.unsqueeze(dim=1) * feature_right_up)
        final_feature = rearrange(final_feature, 'b c h w n -> b c (h n) w')  # (batch,channel,height'*sample_number,width')
        result = self.OutConv(final_feature)
        return result

    @staticmethod
    def generate_initial_offset(sample_number):
        """😄
        function     : 根据采样点数生成可变形卷积中每个采样点的初始位置偏移量
        sample_number: 采样点数
        return       : 若采样点数为11:
                       (0,0) (0,1) (0,2)
                       (1,0) (1,1) (1,2)
                       (2,0) (2,1) (2,2)
                       (3,0) (3,1) .....
                       .....
                       最终返回结果为:[0,0,0,1,1,1,2,2,2,3,3,0,1,2,0,1,2,0,1,2,0,1]
                       形状为:(batch,sample_number*2,1,1)
                       [0,0:sample_number,0,0]==>所有横坐标的偏移量(从左至右、从上至下)
                       [0,sample_number:,0,0] ==>所有纵坐标的偏移量(从左至右、从上至下)
        """
        base_int = round(math.sqrt(sample_number))  # 计算采样点数(sample_number)的平方根并四舍五入,得到网格的列数(或行数)
        row_number = sample_number // base_int
        mod_number = sample_number % base_int
        # base_int=N
        # pos_x                       pos_y
        # (N,N)                       (N,N)
        # 0 0 0 0 0 ... 0 0 0 0 0     0 1 2 3 4 ... N
        # 1 1 1 1 1 ... 1 1 1 1 1     0 1 2 3 4 ... N
        # . . . . . ... . . . . .     0 1 2 3 4 ... N
        # N N N N N ... N N N N N     0 1 2 3 4 ... N
        pos_x, pos_y = torch.meshgrid(torch.arange(0, row_number), torch.arange(0, base_int), indexing='ij')
        pos_x = torch.flatten(pos_x)  # (N^2,)
        pos_y = torch.flatten(pos_y)  # (N^2,)
        if mod_number > 0:
            mod_pos_x, mod_pos_y = torch.meshgrid(torch.arange(row_number, row_number + 1), torch.arange(0, mod_number), indexing='ij')
            mod_pos_x = torch.flatten(mod_pos_x)
            mod_pos_y = torch.flatten(mod_pos_y)
            pos_x, pos_y = torch.cat((pos_x, mod_pos_x)), torch.cat((pos_y, mod_pos_y))
        position = torch.cat([pos_x, pos_y], 0)
        position = position.view(1, 2 * sample_number, 1, 1)
        return position

    def generate_grid(self, height, width, sample_number, dtype):
        """😄
        function     : 生成初始采样点的位置网格,这些初始采样点的位置将用于后续的可变形卷积操作
        height       : 图像高度
        width        : 图像宽度
        sample_number: 采样点数量
        dtype        : 数据类型
        return       : (batch,sample_number*2,height,width)
        """
        # position_x                  position_y
        # (height,width)              (height,width)
        # 0 0 0 0 0 ... 0             0 1 2 3 4 ... N
        # 1 1 1 1 1 ... 1             0 1 2 3 4 ... N
        # . . . . . ... .             . . . . . ... .
        # N N N N N ... N             0 1 2 3 4 ... N
        # 生成网格
        position_x, position_y = torch.meshgrid(torch.arange(0, height * self.stride, self.stride),
                                                torch.arange(0, width * self.stride, self.stride), indexing='ij')
        # (1):形状维度扩充==>(1,1,height,width) (2):维度重复==>(1,sample_number,height,width)
        position_x = torch.flatten(position_x).view(1, 1, height, width).repeat(1, sample_number, 1, 1)
        position_y = torch.flatten(position_y).view(1, 1, height, width).repeat(1, sample_number, 1, 1)
        position = torch.cat([position_x, position_y], 1).type(dtype)
        return position

    def get_position(self, offset, dtype):
        """😄
        function: 结合初始采样点位置、固定的偏移量和预测的偏移量,生成最终的采样点位置
        offset  : (batch,sample_number,height,width)
        dtype   : 数据类型(用于确保所有张量的类型一致)
        return  : (batch,2*sample_number,height,width)
        """
        sample_number, height, width = offset.size(1) // 2, offset.size(2), offset.size(3)
        position = self.generate_grid(height, width, sample_number, dtype)  # (batch,sample_number*2,height,width)
        # (batch,sample_number*2,height//stride,width//stride)+(batch,sample_number*2,1,1)+(batch,sample_number*2,height,width)
        # 卷积核采样点位置=卷积核左上顶点位置+固有偏移量+可学习的偏移量
        sample_position = position + self.initial_offset + offset
        return sample_position

    @staticmethod
    def extract_feature(input_data, position, sample_number):
        """😄
        function     : 从输入特征图input_data中根据给定的采样点位置position提取特征.
        input_data   : (batch,channel,height,width),最原始的输入数据(未经过下采样)
        position     : (batch,height',width',sample_number*2)采样点坐标
        sample_number: 采样点数
        return       : (batch,channel,height',width',sample_number)
        """
        batch, height, width, _ = position.size()
        padded_width = input_data.size(3)
        channel = input_data.size(1)
        input_data = input_data.contiguous().view(batch, channel, -1)  # (batch,channel,height*width)
        # 当想要根据给定的x和y坐标从特征图中提取特定位置的值时,需要将这些坐标转换为一维索引,以便可以直接使用它们来索引张量.这个一维索引可以通过将,坐标乘以特征图
        # 的宽度(padded_width)再加上y坐标来计算得到.  为什么乘以padded_width?  ==> 乘以padded_width(特征图的宽度)是为了将x坐标转换为正确的一维索引
        # 这是因为在二维数组(或矩阵)中,任何给定的行中的列索引都是从0开始的,并且每一行都有相同的列数(即宽度)因此要找到特定元素在一维数组中的位置,需要先确定它所在的列(通过乘以宽度)
        # 然后加上它在该列中的行偏移量(y坐标)
        index = position[..., :sample_number] * padded_width + position[..., sample_number:]  # (batch,height',width',sample_number)
        # (batch,channel,height'*width'*sample_number)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, channel, -1, -1, -1).contiguous().view(batch, channel, -1)
        # 根据索引从输入张量中收集元素
        gather_feature = input_data.gather(dim=-1, index=index).contiguous().view(batch, channel, height, width, sample_number)
        return gather_feature


class AblationS2Net(nn.Module):
    """🔥🚀🌟🎯✨📈📉⚡⭐
    Smooth Gradients Network
    Description: This is My model!!!❤️🧡💛💚💙💜🤎🖤🤍💕💞💓💗💖💘💝💟
    """

    def __init__(self, in_channel, embed, num_class, height, width, ablation, filter_mode='adaptive'):
        """😄
        function  : Initialize
        in_channel: 输入通道数
        embed     : 嵌入维度
        num_class : 类别个数
        height    : 图像高度
        width     : 图像宽度
        """
        super().__init__()
        self.embed = embed
        self.Stem = nn.Sequential(nn.Conv2d(in_channel, embed, kernel_size=1, padding=0), nn.GELU())
        self.Pyramid = PyramidModule(in_channel=embed, height=height, width=width, embed=embed, filter_mode=filter_mode, ablation=ablation)
        self.MLP = nn.Sequential(nn.Conv2d(embed, 32, 5, 1, 2),
                                 nn.Conv2d(32, num_class, kernel_size=3, padding=1, stride=1), )

    def forward(self, input_data):
        """😄
        function  : Forward
        input_data: (1,channel,height,width)
        return    : (1,num_class,height,width)
        """
        stem_data = self.Stem(input_data)
        pyramid = self.Pyramid(stem_data)
        result = self.MLP(pyramid)  # (1,num_class,height,width)
        # result = func.softmax(result, dim=1)   # 添加这一行训练轮数设置为1000
        return result


if __name__ == '__main__':
    print("5211314")
