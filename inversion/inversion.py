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

input_folder_path = 'D:\Desktop\study document in postgraduate\\010projects\\001CNNtest\\input_tif'
output_folder_path = 'D:\Desktop\study document in postgraduate\\010projects\\001CNNtest\\output_tif'
file_list = glob.glob(f'{input_folder_path}/*.tif')

for file in file_list:
    with rasterio.open(file) as src:
        image = src.read()
        meta = src.meta
        mask = ((image[0, :, :] >= -67) & (image[0, :, :] <= 1000) &
                (image[4, :, :] > 0) & (image[4, :, :] <= 1062) & (image.sum(axis=0) != 0))
        meta.update(dtype=rasterio.float32, count=1)
        file_list_basename = file.split('\\')[-1]
        output_file = f'{output_folder_path}/processed_{file_list_basename}'
        with rasterio.open(output_file, 'w', **meta) as dst:
            new_layer = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
            for i in range(image.shape[1]):
                for j in range(image.shape[2]):
                    if not mask[i, j]:
                        new_layer[i, j] = np.nan
                        continue
                    pixel = image[:, i, j]
                    pixel_normalized = scaler.transform([pixel])[0]
                    prediction = model.predict(pixel_normalized.reshape(1, -1))
                    new_layer[i, j] = prediction[0, 0]
            dst.write(new_layer, 1)
print("All Done")
