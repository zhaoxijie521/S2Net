# 在VSCode中运行取消下面注释、在Pycharm中则需要注释
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path
from Models.RCPNet.RCPNet import RCPNet
from Models.S3ANet.S3ANet import S3ANet
from Models.SACNet.SACNet import SACNet
from Models.S2Net.S2Net import S2Net
from Models.S2Net.Ablations import AblationS2Net
from Models.WFSS.WFSS import WFSS
from Models.CEGCN.CeGCN import CeGCN
from Models.SSFCN.SSFCN import SsFCN
from Models.WFCG.WFCG import WFCG
from Utils.Utils import show_env_message, LoadData, Receiver, parsing_json_files
from Config import data_names, train_numbers, fgsm_epsilons, pgd_epsilons, cw_epsilons, load_configuration, except_label, against_train

all_models = {"S2Net": S2Net, "S3ANet": S3ANet, "SACNet": SACNet, "RCPNet": RCPNet, "WFSS": WFSS, "Baseline": AblationS2Net,
              "Baseline+SGS": AblationS2Net, "Baseline+SAC": AblationS2Net, "Baseline+FDGP": AblationS2Net, "Baseline+FDGP+SAC": AblationS2Net,
              "Baseline+SGS+FDGP": AblationS2Net, "Baseline+SGS+FDGP+SAC": AblationS2Net, "Baseline+SGS+SAC": AblationS2Net, "CeGCN": CeGCN, "WFCG": WFCG,
              "SsFCN": SsFCN}

if __name__ == '__main__':
    # 配置信息(通过设置以下9个变量的值更改程序训练结果)🌎💞🤍💜🧮⭐🚀🌟
    # ********************************************************************************
    data_index = 1  # 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SA)
    model_name = "S2Net"
    is_save_attack_data = False  # 是否保存对抗样本以及噪声数据
    is_save_weight_data = False  # 是否保存模型权重
    is_save_json_data = True  # 是否覆盖之前的训练结果
    number = 10  # 训练、攻击的次数
    epsilon = 0.05  # 指定扰动幅度,用于在训练结束后打印对应扰动幅度的对抗样本测试结果
    against_train_epsilon = 0.01  # 对抗训练的扰动幅度
    against_train_epoch = 50
    # ********************************************************************************
    show_env_message()  # 显示环境信息
    script_path = Path(__file__).resolve().parent
    data_name = data_names[data_index]
    train_number = train_numbers[data_index]
    data_path = rf"{script_path}/DataSet/{data_name}"  # 加载数据基本信息
    select_model = all_models[model_name]  # 加载模型实例
    use_against_train = against_train[model_name]  # 是否采取对抗训练
    # 训练、测试、攻击模型
    test_result, attack_example_result = [], {}
    receiver = Receiver(model_name=f"{model_name}", save_path=fr"{script_path}/Result", data_name=data_name)
    for run in range(1, number + 1):
        seed = 202500 + run
        # 模型配置参数,加载模型实例
        model_param, train_setting = load_configuration(data_index=data_index, model=model_name, seed=seed)
        loader = LoadData(data_name=data_name, data_path=data_path, is_print=False)
        image, label_train, label_test = loader.split_data(seeds=seed, number=train_number, except_label=except_label, rate=None)
        model = select_model(**model_param)
        receiver.reset(model=model, except_label=except_label, settings=train_setting)  # 重置接收器
        receiver.train_model(train_image=image, train_label=label_train, show_accuracy=True, index=run, is_save=is_save_weight_data)
        clean_result = receiver.test_model(test_image=image, test_label=label_test, index=run)
        if use_against_train:
            receiver.against_train_model(train_image=image, train_label=label_train, epochs=against_train_epoch, epsilon=against_train_epsilon, index=run,
                                         is_save=is_save_weight_data)
        attack_result = receiver.attack_model(image, label_test, except_label, run, is_save=is_save_attack_data, fgsm_epsilons=fgsm_epsilons,
                                              cw_epsilons=cw_epsilons, pgd_epsilons=pgd_epsilons)
        test_result.append(clean_result)  # 保存测试结果
        attack_example_result[f"{run}"] = attack_result  # 保存攻击结果
    # 将结果保存至JSON文件
    if is_save_json_data:
        with open(fr"{script_path}/Result/Json/{model_name}-{data_name}.json", "w", encoding="utf-8") as file:
            json.dump(attack_example_result, file, ensure_ascii=False, indent=4)
        # 打印本次整体的对抗样本测试结果
        parsing_json_files(data_name=data_name, model_name=model_name, epsilon=epsilon)
