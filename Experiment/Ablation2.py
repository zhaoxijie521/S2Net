import torch
import numpy as np
from pathlib import Path
from Utils.Utils import AnalysisModel, show_mean_var
from Config import generate_adv_example, load_configuration
from Config import data_names, except_label, blue, reset, green, red

script_path = Path(__file__).resolve().parent.parent


def important_notes():
    print(f"本程序用于分析BatchNorm和SAC模块的结果对比🐳")
    print(fr"在运行此程序生成对抗样本之前应确保==>{blue}{script_path.parent}/Result/Weight{reset}文件夹中存在相应的权重文件!!!")
    print(fr"若不存在权重文件,可运行==>{green}{script_path.parent}/TrainModel.py{reset}文件生成相应的权重文件.")
    print(f"当模型权重、数据集或攻击方法发生变更时,需同步更新{green}show_mean_var(){reset}函数的{blue}region{reset}参数值;"
          f"或者将{red}is_ax_ins{reset}统一置为{blue}False{reset},以禁用局部放大图显示")


if __name__ == '__main__':
    # ********************************************************************
    data_index = 1  # 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SA)
    model_index = 0  # 设置其数值,分别为:0⇒Baseline、1⇒Baseline+SAC
    types = "output"  # 设置其值⇒['input','output']
    save_type = "pdf"  # 设置其值⇒['png','pdf']
    method = "FGSM"  # 设置其值⇒['FGSM'、'PGD'、'CW']
    # ********************************************************************
    important_notes()
    # 基础设置
    model1, model2 = "Baseline", "Baseline+SAC"
    model_name = [model1, model2][model_index]
    # 可以运行程序⇒(打印模型结构.py)输出模型的结构
    target_layer = {model1: "Pyramid.PreModule3[1].BatchNorm", model2: "Pyramid.PreModule3[1]"}[model_name]
    data_name = data_names[data_index]
    file_name = [model1, model2][model_index]
    is_ax_ins = {model1: False, model2: True}[model_name]  # noqa
    model_path = fr"{script_path}/Result/Weight/{model_name}/{data_name}/1.pt"  # 模型权重地址
    # 加载配置信息
    _, train_setting = load_configuration(data_index=data_index, model=model_name)
    # 获取对抗样本
    adv_datas = {}
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for epsilon in epsilons:
        adv_image = generate_adv_example(data_name=data_name, model_name=model_name, epsilon=epsilon, ignore_label=except_label,
                                         train_setting=train_setting, method=method)  # 对抗样本(channel,height,width)
        adv_datas[f"{epsilon}"] = adv_image
    # 将对抗样本依次输入至模型中得到其中间特征对比
    analysis = AnalysisModel(weight_path=model_path, except_label=except_label)
    feature_means, feature_vars = [], []
    for epsilon in epsilons:
        if types not in ['input', 'output']:
            raise ValueError("mode类型写错辣!!!")
        mean, var = analysis.get_feature_mean_var(target_layer=target_layer, adv_image=adv_datas[f"{epsilon}"], mode=types)
        feature_means.append(mean)
        feature_vars.append(var)
        torch.cuda.empty_cache()
    feature_means = np.array(feature_means)
    feature_vars = np.array(feature_vars)
    means_labels = epsilons
    vars_labels = epsilons
    # 获取BatchNorm的运行统计量
    if model_index == 0 and types == "input":  # 包含BatchNorm层的模型
        running_mean, running_var, _, _ = analysis.get_batch_norm_params(target_layer=target_layer)
        feature_means = np.vstack((feature_means, running_mean.reshape(1, -1)))
        feature_vars = np.vstack((feature_vars, running_var.reshape(1, -1)))
        means_labels = means_labels + [r'$\mu_{global}$']
        vars_labels = vars_labels + [r'$\sigma^2_{global}$']
    if save_type not in ['png', 'pdf']:
        raise ValueError("结果保存类型写错辣!!!")
    file_type = "输出" if types == "output" else "输入"
    show_mean_var(lines_data=feature_means, line_labels=means_labels, region=[38.999995867, 39.000006329, 0.0319216139, 0.0319218149], location=(0.5, 0.05, 0.3, 0.4),
                  connection_position=(1, 2), save_path=fr"{script_path}/Result/FeatureDrift", file_name=f"{file_name}-{file_type}特征均值.{save_type}",
                  is_ax_ins=is_ax_ins, is_legend=False)
    show_mean_var(lines_data=feature_vars, line_labels=vars_labels, region=[5.98011, 6.01337, 2.27835341, 2.27839725], location=(0.4, 0.05, 0.3, 0.4),
                  connection_position=(1, 3), save_path=fr"{script_path}/Result/FeatureDrift", file_name=f"{file_name}-{file_type}特征方差.{save_type}",
                  is_ax_ins=is_ax_ins, is_legend=False)
