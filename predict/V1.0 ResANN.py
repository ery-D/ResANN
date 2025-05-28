import pandas as pd
from keras.layers import Input, Dense, BatchNormalization, ReLU, Add, Attention, Dropout, Activation
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error, \
    mean_squared_log_error, median_absolute_error
import numpy as np
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
df = pd.read_excel('ToTalNdata.xlsx')
# df = df.drop(df.index[0])
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def residual_block(x, units):
    y = Dense(units, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    y = Dense(units, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(y)
    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

def adjusted_mean_squared_error(y_true, y_pred):
    weights = 1.2 * tf.exp(y_true / tf.reduce_max(y_true))
    weighted_mse = tf.reduce_mean(weights * tf.square(y_true - y_pred))
    l2 = 0.01
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
    total_loss = weighted_mse + l2 * l2_loss
    return total_loss

def build_model(input_shape, units, dropout_rate):
    inputs = Input(shape=(input_shape,))
    x = Dense(units, activation='relu')(inputs)
    x = residual_block(x, units)
    x = Dropout(dropout_rate)(x)
    x = residual_block(x, units)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=adjusted_mean_squared_error, optimizer=Adam(learning_rate=0.0005))
    return model

model = build_model(X_train_scaled.shape[1], units=64, dropout_rate=0.1)
early_stopping_monitor = EarlyStopping(patience=20)
model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=0,
          callbacks=[early_stopping_monitor])

y_train_pred = model.predict(X_train_scaled)
# 计算指标
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
explained_variance_train = explained_variance_score(y_train, y_train_pred)
max_error_value_train = max_error(y_train, y_train_pred)
msle_train = mean_squared_log_error(y_train, y_train_pred)
median_ae_train = median_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
print(f'Train MSE: {mse_train}')
print(f'Train R2: {r2_train}')
print(f'Train MAE: {mae_train}')
print(f'Train Explained Variance Score: {explained_variance_train}')
print(f'Train Max Error: {max_error_value_train}')
print(f'Train Mean Squared Logarithmic Error: {msle_train}')
print(f'Train Median Absolute Error: {median_ae_train}')
print(f'Train MAPE: {mape_train}%')

y_test_pred = model.predict(X_test_scaled)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
explained_variance_test = explained_variance_score(y_test, y_test_pred)
max_error_value_test = max_error(y_test, y_test_pred)
msle_test = mean_squared_log_error(y_test, y_test_pred)
median_ae_test = median_absolute_error(y_test, y_test_pred)
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print(f'Test MSE: {mse_test}')
print(f'Test R2: {r2_test}')
print(f'Test MAE: {mae_test}')
print(f'Test Explained Variance Score: {explained_variance_test}')
print(f'Test Max Error: {max_error_value_test}')
print(f'Test Mean Squared Logarithmic Error: {msle_test}')
print(f'Test Median Absolute Error: {median_ae_test}')
print(f'Test MAPE: {mape_test}%')

now = datetime.now()
time_str = now.strftime("%m-%d_%H-%M-%S")
joblib.dump(scaler, 'scaler_TN.save')
model.save(r'my_model_' + time_str +'.h5')
print("模型已保存为 my_model")
print("模型已保存为 model.h5")

y_train_actual = [y_train]
y_test_actual = [y_test]
y_train_actual = np.array(y_train_actual).flatten()
y_train_pred = np.array(y_train_pred).flatten()
y_test_actual = np.array(y_test_actual).flatten()
y_test_pred = np.array(y_test_pred).flatten()
slope_train, intercept_train = np.polyfit(y_train_actual, y_train_pred, 1)
slope_test, intercept_test = np.polyfit(y_test_actual, y_test_pred, 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train_actual, y_train_pred, color='blue', label='Train')
plt.plot(y_train_actual, intercept_train + slope_train * y_train_actual, 'r-',label=f'Fit Line (Slope: {slope_train:.2f})')
plt.plot([np.min(y_train_actual), np.max(y_train_actual)], [np.min(y_train_actual), np.max(y_train_actual)], 'k--',lw=2, label='y=x line')
plt.text(0.02, 0.8, f'Train: $R^2$={r2_train:.2f} MSE={mse_train:.2f}', transform=plt.gca().transAxes)
plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, y_test_pred, color='orange', label='Test')
plt.plot(y_test_actual, intercept_test + slope_test * y_test_actual, 'r-', label=f'Fit Line (Slope: {slope_test:.2f})')
plt.plot([np.min(y_test_actual), np.max(y_test_actual)], [np.min(y_test_actual), np.max(y_test_actual)], 'k--', lw=2,label='y=x line')
plt.text(0.02, 0.8, f'Test: $R^2$={r2_test:.2f} MSE={mse_test:.2f}', transform=plt.gca().transAxes)
for i in [1, 2]:
    plt.subplot(1, 2, i)
    plt.legend()
    plt.xlabel('TN measured')
    plt.ylabel('TN predicted')
plt.tight_layout()
model_plot = f'1.png'
plt.savefig(model_plot, dpi=300, bbox_inches='tight')
plt.show()
