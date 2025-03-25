import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import load

# 假設你已經載入並預處理 WESAD 數據
# X 為 (samples, timesteps, features)，這裡以 EDA 信號為例
# y 為壓力狀態類別


# 載入資料
wesad = load.WESAD()
df = wesad.separate_and_feature_extract(sample_n=70000)

y = df['label'].to_numpy()-1
X = df.drop(['label','subject'],axis=1)
X = X.to_numpy()

#X = X.reshape(X.shape[0], -1)  # 先展平做標準化
X = X.reshape(*X.shape, 1)  # 轉回 CNN 需要的格式

# 分割訓練測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立 1D CNN 模型
def build_cnn(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv1D(2, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(2, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# 初始化模型
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y))
model = build_cnn(input_shape, num_classes)

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
epochs = 20
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# 評估模型
loss, acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {acc * 100:.2f}%')
