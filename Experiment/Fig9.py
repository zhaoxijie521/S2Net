from pathlib import Path
import scipy.io as sio
from Utils.Utils import set_matplotlib
from Config import reset, green, blue, data_names, load_configuration, generate_adv_example, image_keys, except_label
from Utils.Utils import LoadData, draw_ablation_line

script_path = Path(__file__).resolve().parent  # 当前文件所在文件夹


def important_notes():
    print(fr"在运行此程序生成对抗样本之前应确保==>{blue}{script_path.parent}/Result/Weight{reset}文件夹中存在相应的权重文件!!!")
    print(fr"若不存在权重文件,可运行==>{green}{script_path.parent}/TrainModel.py{reset}文件生成相应的权重文件.")


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
    clean_data = sio.loadmat(fr"{script_path.parent}/DataSet/{data_names}/{data_names}.mat")[image_keys[data_names]]
    clean_data = LoadData.norm_data(clean_data).mean(axis=(0, 1))
    for models in model_names:
        # 加载训练配置
        _, train_settings = load_configuration(data_index=data_index, model=models, seed=5211314)
        adv_data = generate_adv_example(data_name=data_names, model_name=models, epsilon=0.5, ignore_label=except_label, train_setting=train_settings, method='PGD')
        adv_data = adv_data.transpose(1, 2, 0).mean(axis=(0, 1))
        draw_ablation_line(clean_data=clean_data, adv_data=adv_data, data_name=data_names, module_name=models,
                           save_path=fr"{script_path.parent}/Result/Fig9", save_type=save_type)
    print(fr"结果保存至:{blue}{script_path.parent}/Result/Fig9/{data_names}{reset}文件夹中")
