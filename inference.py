import os

import torch
import numpy as np

from utils.getDevice import getDevice
from utils.getData import loadAudio
from utils.getConfig import getConfig

config = getConfig("config.yaml")
save_model = str(config['inference']['save_model'])
trained_model = str(config['inference']['trained_model'])
audio_path = str(config['inference']['audio_path'])
genre_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def inference():
    device = getDevice()
    # 加载模型
    model = torch.jit.load(os.path.join(save_model, trained_model))
    model.to(device)
    model.eval()
    print("Using model: " + trained_model)
    # 加载音频
    spec_mel = loadAudio(audio_path)
    spec_mel = spec_mel[np.newaxis, :]
    spec_mel = torch.tensor(spec_mel, dtype=torch.float32, device=device)
    print("Audio path: " + audio_path)
    # 执行预测
    pred = model(spec_mel)
    pred = pred.data.cpu().numpy()[0]
    pred = np.argmax(pred)
    print("Inference result: " + genre_list[pred])


if __name__ == '__main__':
    inference()
