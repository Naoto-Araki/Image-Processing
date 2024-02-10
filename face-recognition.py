import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

cap = cv2.VideoCapture(0)

face_cascade_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_file)

#作成した学習モデルを読み込む
clf = pickle.load(open("face_model.sav","rb"))

end_flag, frame = cap.read()
height, width, channels = frame.shape

cv2.namedWindow('image')

while end_flag == True:

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_list = face_cascade.detectMultiScale(frame_gray, minSize=(30, 30))

    for (x, y, w, h) in face_list:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)

        #検出した顔を学習モデルによって、判別する
        #検出した顔部分だけを切り取る
        cut_img = frame[y: y + h, x: x + w]
        
        #顔判定用のリストを作る
        recog =[]

        #リアルタイムに認識した顔を recognition_sav.py の時にリサイズした50×50に合わせる
        recog_img = cv2.resize(cut_img,(50,50),interpolation=cv2.INTER_LINEAR)

        #グレースケールに変換
        recog_gray = cv2.cvtColor(recog_img, cv2.COLOR_BGR2GRAY)

        #1次元の行列に変換
        recog_gray = np.array(recog_gray,"uint8").flatten()

        #顔判定用の箱に入れる
        recog.append(recog_gray)
        
        #行列に変換
        recog = np.array(recog)

        #予測実行
        pred = clf.predict(recog)

        #予測した画像のラベルから表示させる名前を決定する
        if pred == "〇〇〇":
            name = "〇〇〇"
            cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
        elif pred == "△△△":
            name = "△△△"
            cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
        elif pred == "□□□":
            name = "□□□"
            cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)  

    cv2.imshow('image', frame)

    key = cv2.waitKey(30)
    if key == 27 or key == ord('q'):
        break

    end_flag, frame = cap.read()

cv2.destroyAllWindows()
cap.release()
