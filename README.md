# 顔検出と個人識別
opencvを使って、顔検出し、個人識別を行うシステムの実装に取り組む。

## 環境構築
```shell
pip install opencv pillow scikit-learn numpy matplotlib
```

## scikit-learnで学習モデルを作成する
学習モデルを作成するため、個人識別したい人物の写真を各200枚程度収集し、その人物の名前をラベルにする。\
その写真に対して、顔部分のみを取得するために`face-gain.py`を実行する。\
そこで得られた画像から、顔部分以外が検出されている画像を削除して、それを訓練データとして使用する。

学習モデルを作成するために、scikit-learnを用いて、SVMを実装した`recognition_sav.py`を実行する。

## パソコンの内蔵カメラでリアルタイムに顔検出し、個人識別を行う
`recognition_sav.py`を実行して得られた学習モデル`face_model.sav`を用いて、`face-recognition.py`を実行する。\
実行後、カメラが起動され、顔検出し、赤い枠線部分にその学習した人物の名前が表示される。

### 参考文献
https://qiita.com/daiarg/items/3ea91b08f0d1cb5bfc61#2-%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E4%BD%9C%E6%88%90

