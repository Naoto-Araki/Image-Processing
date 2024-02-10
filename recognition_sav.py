import cv2
import numpy as np
from PIL import Image
import os
from sklearn import svm
import pickle

#訓練データ
images = []
labels = []

#pathの場所は自分で指定する
path = "img"

for i in os.listdir(path):
    image_path = os.path.join(path, i)

    # 画像を取り込む
    img = cv2.imread(image_path)
    # ImageをNumpy配列に変換
    img_np = np.array(img)

    # 画像を同じサイズにリサイズ
    img_resized = cv2.resize(img_np, (50, 50), interpolation=cv2.INTER_LINEAR)

    # 取り込んだ画像をグレースケールに変更
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # imgを1次元配列に変換
    img_flattened = img_gray.flatten()

    # images[]にimgを格納
    images.append(img_flattened)

    #ファイル名からラベルを取得
    labels.append(str(i[0:3]))

#行列に変換
labels = np.array(labels)
images = np.array(images)

#svmの変換器を作成
clf = svm.LinearSVC(dual=False, max_iter=1000)
#学習
clf.fit(images,labels)

#学習モデルを保存する
filename = "face_model.sav"
pickle.dump(clf,open(filename,"wb"))
