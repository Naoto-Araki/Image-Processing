import cv2
from PIL import Image
import os

face_cascade_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_file)

#input_path, out_pathの場所は自分で指定する
input_path = "img_train"
output_path = "img"

for i in os.listdir(input_path):
    image_path = os.path.join(input_path, i)

    # 画像を取り込む
    img = cv2.imread(image_path)

    # 画像が正しく読み込まれたかの確認
    if img is not None:
        # 取り込んだ画像をグレースケールに変更
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 顔検出
        face_list = face_cascade.detectMultiScale(img_gray, minSize=(30, 30))

        for (x, y, w, h) in face_list:
            # 顔の部分を切り出し
            face_image = img[y : y + h, x : x + w]

            # 出力先のファイルパスを生成
            output_file_path = os.path.join(output_path, i)

            # 顔の部分を保存
            cv2.imwrite(output_file_path, face_image)

            print(f"顔を切り出して {output_file_path} に保存しました。")

    else:
        print(f"画像 {image_path} を正しく読み込めませんでした。")
