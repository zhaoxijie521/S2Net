import numpy as np
from Utils.Utils import set_matplotlib
from Config import *
from Utils.Utils import LoadData, draw_ablation_line

script_path = Path(__file__).resolve().parent  # 当前文件所在文件夹


def important_notes():
    print(fr"在运行此程序生成对抗样本之前应确保==>{blue}{script_path.parent}\Result\Weight{reset}文件夹中存在相应的权重文件!!!")
    print(fr"若不存在权重文件,可运行==>{green}{script_path.parent}\TrainModel.py{reset}文件生成相应的权重文件.")


def generate_adv_example(data_name: str, model_name: str, epsilon: float, ignore_label: int, train_setting: dict, method="PGD") -> np.ndarray:
    """
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
    # 加载图像、标签数据
    file_name = file_names[data_name]
    image = sio.loadmat(fr"{script_path.parent}\DataSet\{data_name}\{data_name}.mat")[image_keys[data_name]]
    label = sio.loadmat(fr"{script_path.parent}\DataSet\{data_name}\{file_name}.mat")[label_keys[data_name]]
    # 转为tensor数据
    image = torch.from_numpy(LoadData.norm_data(image).transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)
    label = torch.from_numpy(label[None, :, :]).to(device)
    # 加载模型
    model_path = fr"{script_path.parent}\Result\Weight\{model_name}\{data_name}\1.pt"
    model = torch.load(model_path, weights_only=False).to(device)
    # 设置模型接收器
    receiver = Receiver(model_name=model_name, save_path=rf"{script_path.parent}\Result", data_name=data_name)
    receiver.reset(model=model, except_label=ignore_label, settings=train_setting)
    # 生成对抗样本
    print(f"攻击模型:{model_name}")
    # 对抗样本形状==>{"method-epsilon":numpy.array(chanel,height,width),...,"method-epsilon":numpy.array(chanel,height,width)}
    if method == "PGD":
        adv_image = receiver.pgd_attack(image=image, true_label=label, root_dir="", epsilons=[epsilon])
    elif method == 'FGSM':
        adv_image = receiver.target_fgsm_with_norm(image=image, label=label, root_dir="", epsilons=[epsilon])
    else:
        adv_image = receiver.cw_attack(image=image, true_label=label, root_dir="", epsilons=[epsilon])
    return adv_image[fr"{method}-{epsilon}"]


if __name__ == '__main__':
    # 针对指定数据集,绘制各模型干净样本与对抗样本的光谱均值对比曲线🐳
    # ********************************************************************
    data_index = 2  # 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SV)
    model_names = ["Baseline", "Baseline+SGS+FDGP", "Baseline+SGS+SAC", "Baseline+FDGP+SAC", "Baseline+SGS+FDGP+SAC"]
    save_type = 'pdf'  # 图片保存类型
    data_names = data_names[data_index]
    # ********************************************************************
    important_notes()
    set_matplotlib()
    # 加载干净样本数据
    clean_data = sio.loadmat(fr"{script_path.parent}\DataSet\{data_names}\{data_names}.mat")[image_keys[data_names]]
    clean_data = LoadData.norm_data(clean_data).mean(axis=(0, 1))
    for models in model_names:
        # 加载训练配置
        _, train_settings = load_configuration(data_index=data_index, model=models)
        adv_data = generate_adv_example(data_name=data_names, model_name=models, epsilon=0.5, ignore_label=except_label, train_setting=train_settings, method='PGD')
        adv_data = adv_data.transpose(1, 2, 0).mean(axis=(0, 1))
        draw_ablation_line(clean_data=clean_data, adv_data=adv_data, data_name=data_names, module_name=models,
                           save_path=fr"{script_path.parent}\Result\Fig9", save_type=save_type)
    print(fr"结果保存至:{blue}{script_path.parent}\Result\Fig9\{data_names}{reset}文件夹中")
