from Config import data_names, blue, reset, green
from Utils.Utils import plot_model_robustness_curves
from pathlib import Path


def important_notes():
    print(fr"运行此程序需要==>{blue}{Path(__file__).resolve().parent.parent}/Result/Json{reset}地址中存在各个模型的对抗样本测试结果.")
    print(fr"变量{blue}method、epsilons{reset}所指定的数据必须存在于相应的Json文件中")
    print(fr"若不存在可运行==>{green}{Path(__file__).resolve().parent.parent}/ModelTrain.py{reset},并设置覆盖训练结果!!!")


if __name__ == '__main__':
    # 绘制不同模型在不同攻击扰动强度下的整体精度变化曲线🐳
    # ********************************************************************
    data_index = 0  # 设置其数值:0⇒(PU)、1⇒(IP)、2⇒(SA)
    save_type = 'pdf'  # 设置其值⇒png或pdf
    indicators = 'OA'  # 设置其值⇒'OA'、'AA'、'Kappa'
    method = "FGSM"  # 设置其值⇒'FGSM'、'PGD'、'CW'
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # ********************************************************************
    important_notes()
    data_name = data_names[data_index]
    script_path = Path(__file__).resolve().parent.parent
    json_path = fr"{script_path}/Result/Json"
    legend_name = {"MyModel": "$S^{2}Net$", "BANet": "BAN", "S3ANet": r"$S^{3}ANet$", "SACNet": "SACNet", "WFSS": "WFSS",
                   "WFCG": "WFCG", "CeGCN": "CEGCN", "RCPNet": "RCCA", "GSCViT": "GSCViT", "EATN": "EATN"}  # noqa
    save_path = fr"{script_path}/Result/Fig8"
    plot_model_robustness_curves(root_path=json_path, model_name=legend_name, dataset_name=data_name, epsilons=epsilons, indicators=indicators,
                                 method="FGSM", save_path=save_path, save_type=save_type)
