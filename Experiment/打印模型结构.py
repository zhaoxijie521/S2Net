import pprint
import torch
from pathlib import Path
from Config import blue, reset, green, data_names

script_path = Path(__file__).resolve().parent.parent


def important_notes():
    print(f"本程序用于根据模型的权重输出模型的结构.")
    print(fr"在运行此程序生成对抗样本之前应确保==>{blue}{script_path}\Result\Weight{reset}文件夹中存在相应的权重文件!!!")
    print(fr"若不存在权重文件,可运行==>{green}{script_path}\TrainModel.py{reset}文件生成相应的权重文件.")


if __name__ == '__main__':
    # ********************************************************************
    data_index = 1  # 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SV)
    model_index = 1  #
    # ********************************************************************
    model_name = ["Base+SGS+FDGP", "S2Net"][model_index]
    data_name = data_names[data_index]
    important_notes()
    weight_path = fr"{script_path}\Result\Weight\{model_name}\{data_name}\1.pt"
    model = torch.load(weight_path, weights_only=False)
    pprint.pprint(model)
