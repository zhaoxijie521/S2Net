from pathlib import Path
from Config import data_names, blue, green, reset
from Utils.Utils import parsing_json_files

script_path = Path(__file__).resolve().parent.parent


def important_notes():
    print(fr"在运行此程序生成对抗样本之前应确保==>{blue}{script_path}/Result/Json{reset}文件夹中存在相应的数据文件!!!")
    print(fr"若不存在权重文件,可运行==>{green}{script_path}/TrainModel.py{reset}文件生成相应的数据文件.")


if __name__ == '__main__':
    # ********************************************************************
    model_index = 0  # 指定模型
    data_index = 2  # 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SV)
    epsilon = 0.05  # 指定需要打印对抗样本测试精度时对应的扰动幅度
    # ********************************************************************
    important_notes()
    model_name = ["S2Net", "WFCG", "CeGCN", "BANet", "SACNet", "S3ANet", "MHIAFormer", "WFSS", "RCPNet", "GSCViT", "EATN", "DSFormer"][model_index]
    data_name = data_names[data_index]
    parsing_json_files(data_name=data_name, model_name=model_name, epsilon=epsilon)
