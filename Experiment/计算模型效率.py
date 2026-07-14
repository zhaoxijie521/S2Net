import torch
from thop import profile
from pathlib import Path
from Config import data_names, green, blue, reset, red

script_path = Path(__file__).resolve().parent.parent


def important_notes():
    print(f"本程序用于计算每个模型的计算量、参数量.")
    print(fr"在运行此程序生成对抗样本之前应确保==>{blue}{script_path}/Result/Weight{reset}文件夹中存在相应的权重文件!!!")
    print(fr"若不存在权重文件,可运行==>{green}{script_path}/TrainModel.py{reset}文件生成相应的权重文件.")


if __name__ == '__main__':
    # ********************************************************************
    model_names = ["S2Net", "BANet", "RCPNet", "S3ANet", "SACNet", "CeGCN", "WFCG", "WFSS"]
    data_index = 1  # 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SV)
    # ********************************************************************
    important_notes()
    length = max([len(i) for i in model_names])
    data_name = data_names[data_index]
    data = {"PaviaU": torch.randn(1, 103, 610, 340), "Indian_pines_corrected": torch.randn(1, 200, 145, 145),
            "Salinas_corrected": torch.randn((1, 204, 512, 217))}[data_name]
    print(f"在数据集{green}{data_name}{reset}上,每个模型的计算量、参数量结果如下:")
    for model_name in model_names:
        data = data.cuda()
        model_path = fr"{script_path}/Result/Weight/{model_name}/{data_name}/1.pt"
        model = torch.load(model_path, weights_only=False).cuda()
        flops, parameters = profile(model, inputs=(data,), verbose=False)
        print(f"model={model_name: <{length}}  flops={red}{flops / 1e6:10.2f}{reset} parameters={red}{parameters / 1e6:7.2f}{reset}")
        torch.cuda.empty_cache()
