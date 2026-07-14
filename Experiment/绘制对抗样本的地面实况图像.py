# 根据模型的权重攻击模型,产生对抗样本,测试对抗样本的分类结果,并根据分类结果绘制地面实况图像👌
import torch
import scipy.io as sio
from pathlib import Path
from Config import data_names, label_keys, file_names, image_keys, except_label, device, blue, reset, green, load_configuration
from Utils.Utils import draw_ground_truth, LoadData, Receiver

script_path = Path(__file__).resolve().parent.parent


def important_notes():
    print(fr"在运行此程序生成对抗样本之前应确保==>{blue}{script_path.parent}/Result/Weight{reset}文件夹中存在相应的权重文件!!!")
    print(fr"若不存在权重文件,可运行==>{green}{script_path.parent}/TrainModel.py{reset}文件生成相应的权重文件.")


if __name__ == '__main__':
    # 绘制不同攻击方法和扰动幅度下对抗样本预测结果对应的地面实况图像🐳
    # ********************************************************************
    data_index = 2  # 设置其数值,分别为:0⇒(PU)、1⇒(IP)、2⇒(SV)
    model_names = ["WFCG", "CeGCN", "S2Net", "BANet", "RCPNet", "S3ANet", "SACNet", "WFSS"]
    epsilon = 0.05
    method = "FGSM"  # ['FGSM','PGG','CW']
    # ********************************************************************
    data_name = data_names[data_index]
    file_name = file_names[data_name]
    label_key = label_keys[data_name]
    image_key = image_keys[data_name]
    if method not in ['FGSM', 'PGD', 'CW']:
        raise ValueError("攻击方式超出范围辣!!!")
    save_file = {"PaviaU": 'PU', "Indian_pines_corrected": 'IP', "Salinas_corrected": "SV"}[data_name]
    model_types = {"S3ANet": 'General', "S2Net": 'General', "RCPNet": 'RCPNet', "SACNet": 'General', "BANet": 'BANet', "WFSS": "General",
                   "CeGCN": "General", "WFCG": "General"}
    save_path = fr"{script_path}/Result/AdvGT-{method}-{epsilon}/{save_file}"
    for model_name in model_names[:2]:  # 先不跑WFSS(显存不够,须在4090上运行)
        _, train_setting = load_configuration(data_index=data_index, model=model_name, seed=5211314)
        # 加载图像标签数据
        data_path = rf"{script_path}/DataSet/{data_name}"
        image_data = sio.loadmat(fr"{script_path}/DataSet/{data_name}/{data_name}.mat")[image_key]
        image_data = torch.from_numpy(LoadData.norm_data(image_data).transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)
        label_data = sio.loadmat(fr"{script_path}/DataSet/{data_name}/{file_name}.mat")[label_key]
        true_label = torch.from_numpy(label_data[None, :, :]).to(device)
        # 加载模型
        model_path = fr"{script_path}/Result/Weight/{model_name}/{data_name}/1.pt"
        model = torch.load(model_path, weights_only=False).to(device)
        # 创建模型接收器,重置模型接收器
        receiver = Receiver(model_name=f"{model_name}", save_path=rf"{script_path}\Result", data_name=data_name)
        receiver.reset(model=model, except_label=except_label, settings=train_setting)
        print(f"攻击模型==>{blue}{model_name}{reset}")
        if method == "FGSM":
            adv_images = receiver.target_fgsm_with_norm(image=image_data, label=true_label, root_dir="", epsilons=[epsilon])
        elif method == "PGD":
            adv_images = receiver.pgd_attack(image=image_data, true_label=true_label, root_dir="", epsilons=[epsilon])
        else:
            adv_images = receiver.cw_attack(image=image_data, true_label=true_label, root_dir="", epsilons=[epsilon])
        adv_image = torch.from_numpy(adv_images[f"{method}-{epsilon}"]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(adv_image)
        result = torch.argmax((output if model_types[model_name] == "General" else output[0]), dim=1).cpu().numpy()
        draw_ground_truth(true_label=true_label.cpu().numpy(), predict_label=result, dataset_name=data_name, is_show=True,
                          save_path=save_path, file_name=fr"{model_name}-{data_name}")
    print(f"图片已保存至==>{blue}{save_path}{reset}")
