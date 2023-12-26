# Music Genre Classification
___
>A music genre classification model for ten genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock.

References:

https://github.com/xiaoyou-bilibili/voice_recognize

https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch

Please install dependencies first.
```bash
pip install requirements.txt
```
Please install PyTorch corresponding to your own CUDA and Python version.

Please create directories named 'data', 'models' and 'inference'.
## About Training
Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

After downloading, separate the various styles into 'train' and 'test' parts, and place them respectively in 'data/train' and 'data/test' folders.

Directory Structure
```
├── data
  ├── train
    ├──blues
      ├──audio1
      ├──audio2
    ├──classical
      ├──audio1
      ├──audio2
    ...
  ├── test
    ├──blues
      ├──audio1
      ├──audio2
    ├──classical
      ├──audio1
      ├──audio2
    ...
```
Then set the parameters in the 'train' section of the config.yaml file.

Run
```bash
python preprocess.py
python train.py
```
## About Inference
Please set the 'audio_path' in the 'inference' section of config.yaml to the path of the audio file for inference.

Then run
```bash
python inference.py
```
## About Model
Net: Resnet

Loss function: CrossEntropyLoss

Optimizer：Adam
___
>blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rockの10種類の音楽ジャンル分類モデル

このプロジェクトの参考:

https://github.com/xiaoyou-bilibili/voice_recognize

https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch

環境を先に構築してください
```bash
pip install requirements.txt
```
ご自身のCUDAとPythonのバージョンに応じたPyTorchの対応バージョンをインストールしてください

data、modelsとinferenceというディレクトリを作成してください
## 訓練に関して
データセット: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

ダウンロード後、各スタイルを訓練用（train）とテスト用（test）に分け、それぞれをdata/trainとdata/testのフォルダに入れてください

ディレクトリ構造
```
├── data
  ├── train
    ├──blues
      ├──audio1
      ├──audio2
    ├──classical
      ├──audio1
      ├──audio2
    ...
  ├── test
    ├──blues
      ├──audio1
      ├──audio2
    ├──classical
      ├──audio1
      ├──audio2
    ...
```
その後、config.yamlのtrain項目でパラメータを設定してください

実行
```bash
python preprocess.py
python train.py
```
## 推論に関して
config.yamlのinference項目で、推論を行う音楽ファイルのPathをaudio_pathとして設定してください

実行
```bash
python inference.py
```
## モデルに関して
ネット：Resnet

損失関数：CrossEntropyLoss

最適化アルゴリズム：Adam
___
>针对blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock十种风格的音乐风格分类模型

本项目参考:

https://github.com/xiaoyou-bilibili/voice_recognize

https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch

请先安装依赖
```bash
pip install requirements.txt
```
请根据自己的CUDA和Python版本安装对应版本的Pytorch

请新建名为data,models和inference目录
## 关于训练
数据集: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

下载后将各风格分为train和test两部分，分别放入data/train和data/test下

目录结构
```
├── data
  ├── train
    ├──blues
      ├──audio1
      ├──audio2
    ├──classical
      ├──audio1
      ├──audio2
    ...
  ├── test
    ├──blues
      ├──audio1
      ├──audio2
    ├──classical
      ├──audio1
      ├──audio2
    ...
```
然后在config.yaml的train项中设置参数

运行
```bash
python preprocess.py
python train.py
```
## 关于推理
请在config.yaml的inference项中设置audio_path为要推理的音频路径

运行
```bash
python inference.py
```
## 关于模型
网络：Resnet

损失函数：CrossEntropyLoss

优化器：Adam