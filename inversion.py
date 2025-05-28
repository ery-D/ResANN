import numpy as np
import rasterio
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from rasterio.enums import MaskFlags
import glob
import os

model = load_model('model.h5')
scaler = joblib.load('scaler.save')
# 指定包含tif文件的文件夹路径,使用glob搜索该文件夹下所有的tif文件
#### 注意修改图像输入输出文件夹
input_folder_path = 'D:\Desktop\study document in postgraduate\\010projects\\001CNNtest\\input_tif'
output_folder_path = 'D:\Desktop\study document in postgraduate\\010projects\\001CNNtest\\output_tif'
file_list = glob.glob(f'{input_folder_path}/*.tif')

# 对每个图像应用归一化并进行预测
for file in file_list:
    with rasterio.open(file) as src:
        image = src.read()
        # 获取图像的基本信息
        meta = src.meta
        # 检查第一个波段和第五个波段的值是否在指定范围内
        mask = ((image[0, :, :] >= -67) & (image[0, :, :] <= 1000) &
                (image[4, :, :] > 0) & (image[4, :, :] <= 1062) & (image.sum(axis=0) != 0))
        # 更新元数据以保存单波段图像
        meta.update(dtype=rasterio.float32, count=1)
        # 保存处理后的图像
        file_list_basename = file.split('\\')[-1]
        output_file = f'{output_folder_path}/processed_{file_list_basename}'
        with rasterio.open(output_file, 'w', **meta) as dst:
            # 创建一个新的图层来存储预测结果
            new_layer = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
            # 预测
            for i in range(image.shape[1]):
                for j in range(image.shape[2]):
                    if not mask[i, j]:
                        # 如果像元不满足条件，则跳过
                        new_layer[i, j] = np.nan
                        continue
                    # 提取当前像元的8个波段值
                    pixel = image[:, i, j]
                    # 归一化
                    pixel_normalized = scaler.transform([pixel])[0]
                    prediction = model.predict(pixel_normalized.reshape(1, -1))
                    new_layer[i, j] = prediction[0, 0]
            # 写入新图层
            dst.write(new_layer, 1)
print("所有图像处理完成")


'''
import glob
import os
import shutil  # 用于创建目录

input_folder_path = 'D:\\Desktop\\study document in postgraduate\\010projects\\001CNNtest\\input_tif'
output_folder_path = 'D:\\Desktop\\study document in postgraduate\\010projects\\001CNNtest\\output_tif'

file_list = glob.glob(f'{input_folder_path}\\**\\**\\*ed.tif', recursive=True)

for file_path in file_list:
    # 提取相对于 input_folder_path 的子路径
    relative_path = os.path.relpath(file_path, input_folder_path)
    # 构建输出文件的完整路径
    output_file_path = os.path.join(output_folder_path, relative_path)

    # 提取输出文件的目录路径，并创建目录
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # 假设这里进行一些处理...
    # 处理完成后，保存到 output_file_path
    # 示例：shutil.copy(file_path, output_file_path)  # 仅作为示例

    print(f"Processed file saved to: {output_file_path}")

print("所有图像处理完成")


'''