from Utils.Utils import PerturbationSpectrum
from pathlib import Path
from Config import data_names, blue, green, reset, generate_perturbation

script_path = Path(__file__).resolve().parent


def important_notes():
    print(f"对各个模型的对抗扰动进行频谱分析(计算其径向频谱曲线)🐳")
    print(fr"在运行此程序生成对抗样本之前应确保==>{blue}{script_path.parent}/Result/Weight{reset}文件夹中存在相应的权重文件!!!")
    print(fr"若不存在权重文件,可运行==>{green}{script_path.parent}/TrainModel.py{reset}文件生成相应的权重文件.")


if __name__ == '__main__':
    # ********************************************************************
    data_index = 2  # 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SA)
    model_names = ["S3ANet", "SACNet", "RCPNet", "BANet", 'WFSS', "WFCG", "CeGCN", "Baseline", "Baseline+SGS", "Baseline+FDGP+SAC", "S2Net"]
    save_type = "pdf"  # 设置其值⇒['png','pdf']
    method = "PGD"  # 设置其值⇒['FGSM'、'PGD'、'CW']
    epsilon = 0.5
    # ********************************************************************
    important_notes()
    data_name = data_names[data_index]
    # 生成噪声数据
    perturbation_data = generate_perturbation(data_index=data_index, model_names=model_names, method=method, epsilon=epsilon)
    # 每个模型的径向频谱曲线颜色、legend图标标签
    color = {"S2Net": 'black', "BANet": "#4361ee", "RCPNet": "#ff9900", "S3ANet": "#ff66ff", "SACNet": "#008000", "WFSS": "red", "Baseline": "#b5179e",
             "Baseline+SGS": "#9900ff", "Baseline+FDGP+SAC": "#55d691", "CeGCN": "#8b5a2b", "WFCG": "#00bcd4"}
    label = {"S2Net": r"$S^{2}Net$", "BANet": "BAN", "S3ANet": r"$S^{3}ANet$", "SACNet": "SACNet", "WFCG": "WFCG", "WFSS": "WFSS",
             "Baseline": "Baseline", "Baseline+SGS": "Baseline+SGS", "Baseline+FDGP+SAC": "Baseline+FDGP+SAC", "RCPNet": "RCCA", "CeGCN": "CEGCN"}  # noqa
    PerturbationSpectrum.plot_radial_spectrum(dict_data=perturbation_data, dataset=data_name, color=color, label=label,
                                              save_path=fr"{script_path.parent}/Result/PerturbationAnalysis",
                                              remove_dc=True, use_log=True, use_power=True, save_type='pdf')
