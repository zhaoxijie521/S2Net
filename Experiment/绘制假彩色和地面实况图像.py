import os
import scipy.io as sio
from Utils.Utils import LoadData, draw_ground_truth
from Config import data_names, image_keys, label_keys, data_shapes, blue, reset, file_names
from pathlib import Path

if __name__ == '__main__':
    # 根据图像数据绘制假彩色图像🐳
    script_path = Path(__file__).resolve().parent.parent
    save_path = fr"{script_path}\Result\FalseColor+GroundTruth"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for data_name in data_names:
        image_key = image_keys[data_name]
        label_key = label_keys[data_name]
        file_name = file_names[data_name]
        data_shape = data_shapes[data_name]
        image_data = sio.loadmat(fr"{script_path}\Dataset\{data_name}\{data_name}.mat")[image_key]
        LoadData.display_hsi_image(image_data=image_data, data_name=data_name, show=True, save_path=save_path)
        true_label = sio.loadmat(fr"{script_path}\Dataset\{data_name}\{file_name}.mat")[label_key].reshape(1, *data_shape)
        draw_ground_truth(true_label=true_label, predict_label=true_label - 1, dataset_name=data_name, is_show=True, save_path=save_path,
                          file_name=data_name, title=f"数据集{data_name}地面实况图像(Ground Truth Image)")
    print(f"数据的假彩色、地面实况图像已保存至文件夹==>{blue}{save_path}{reset}中")
