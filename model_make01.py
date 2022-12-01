import os #osモジュール（os機能がpythonで扱えるようにする）
import cv2 #画像や動画を処理するオープンライブラリ
import numpy as np #python拡張モジュール
import matplotlib.pyplot as plt#グラフ可視化
import pandas as pd

import tensorflow
from keras.utils import to_categorical #正解ラベルをone-hotベクトルで求める
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D#全結合層、過学習予防、平滑化、インプット
from keras.applications.vgg16 import VGG16 #学習済モデル
from keras.models import Model, Sequential #線形モデル
from keras import optimizers #最適化関数

#画像の格納
path_BI = [filename for filename in os.listdir('/Users/izawayasushinozomi/Desktop/ブルーアイズ') if not filename.startswith('.')] 
path_BM = [filename for filename in os.listdir('/Users/izawayasushinozomi/Desktop/ブラックマジシャン') if not filename.startswith('.')]
path_RI = [filename for filename in os.listdir('/Users/izawayasushinozomi/Desktop/レッドアイズ') if not filename.startswith('.')]
path_GIA =[filename for filename in os.listdir('/Users/izawayasushinozomi/Desktop/暗黒騎士ガイア') if not filename.startswith('.')]
path_MOS = [filename for filename in os.listdir('/Users/izawayasushinozomi/Desktop/究極完全態・グレート・モス') if not filename.startswith('.')]

#50x50のサイズに指定
image_size = 50

#画像を格納するリスト作成
img_BI = []
img_BM = []
img_RI = []
img_GIA = []
img_MOS = []

for i in range(len(path_BI)):
    img = cv2.imread('/Users/izawayasushinozomi/Desktop/ブルーアイズ/' + path_BI[i])
    img = cv2.resize(img,(image_size, image_size))
    img_BI.append(img)

for i in range(len(path_BM)):
    img = cv2.imread('/Users/izawayasushinozomi/Desktop/ブラックマジシャン/' + path_BM[i])
    img = cv2.resize(img,(image_size, image_size))
    img_BM.append(img)

for i in range(len(path_RI)):
    img = cv2.imread('/Users/izawayasushinozomi/Desktop/レッドアイズ/' + path_RI[i])
    img = cv2.resize(img,(image_size, image_size))
    img_RI.append(img)

for i in range(len(path_GIA)):
    img = cv2.imread('/Users/izawayasushinozomi/Desktop/暗黒騎士ガイア/' + path_GIA[i])
    img = cv2.resize(img,(image_size, image_size))
    img_GIA.append(img)

for i in range(len(path_MOS)):
    img = cv2.imread('/Users/izawayasushinozomi/Desktop/究極完全態・グレート・モス/' + path_MOS[i])
    img = cv2.resize(img,(image_size, image_size))
    img_MOS.append(img)

#np.arrayでXに学習画像、ｙに正解ラベルを代入
X = np.array(img_BI + img_BM + img_RI + img_GIA + img_MOS)

#正解ラベルの作成
y =  np.array([0]*len(img_BI) + [1]*len(img_BM) + [2]*len(img_RI) + [3]*len(img_GIA) + [4]*len(img_MOS))

#配列のラベルをシャッフルする
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]

#学習データと検証データを用意
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#正解ラベルをone-hotベクトルで求める
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#モデルの入力画像として用いるためのテンソールのオプション
input_tensor = Input(shape=(image_size, image_size, 3))

#転移学習のモデルとしてVGG16を使用
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

#学習データに対しての水増し
params = {
    'rotation_range': 30,
    'vertical_flip': True,
    'channel_shift_range': 30,
    'width_shift_range':0.05,
    'height_shift_range':0.05,
    'zoom_range': [0.95, 1.05]
}
generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(**params)

train_iter = generator.flow(X_train, y_train)

print(train_iter)
#モデルの定義 *活性化関数relu
#転移学習の自作モデル
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(64, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(32, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))

#入力はvgg.input, 出力はtop_modelにvgg16の出力を入れたもの
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

#vgg16による特徴抽出部分の重みを固定（以降に新しい層（top_model)が追加）
for layer in model.layers[:19]:
   layer.trainable = False

#コンパイル
model.compile(loss='categorical_crossentropy',
             optimizer="adam",
             metrics=['accuracy'])

# 学習の実行
history01 = model.fit(X_train, y_train, batch_size=32, epochs=80, verbose=1, validation_data=(X_test, y_test))

#acc, val_accのプロット
plt.plot(history01.history["accuracy"], label="accuracy", ls="-", marker=".")
plt.plot(history01.history["val_accuracy"], label="val_accuracy", ls="-", marker=".")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

#loss, val_lossのプロット
plt.plot(history01.history["loss"], label="loss", ls="-", marker=".")
plt.plot(history01.history["val_loss"], label="val_loss", ls="-", marker=".")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

# 学習の実行
history02 = model.fit(train_iter, batch_size=32, epochs=80, verbose=1, validation_data=(X_test, y_test))

#acc, val_accのプロット
plt.plot(history02.history["accuracy"], label="accuracy", ls="-", marker=".")
plt.plot(history02.history["val_accuracy"], label="val_accuracy", ls="-", marker=".")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

#loss, val_lossのプロット
plt.plot(history02.history["loss"], label="loss", ls="-", marker=".")
plt.plot(history02.history["val_loss"], label="val_loss", ls="-", marker=".")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

#精度の評価
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#モデルを保存
model.save("my_new_model.h5")
