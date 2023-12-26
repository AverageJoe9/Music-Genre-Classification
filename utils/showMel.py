import librosa.feature
import numpy as np
import matplotlib.pyplot as plt

wav, sr = librosa.load("../inference/桐谷華 - レイル・ロマネスク ハチロクver.mp3", sr=16000)
melspec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=512, n_mels=257)
logmelspec = librosa.power_to_db(melspec, ref=np.max)
# D是频率，N是长度
D, N = logmelspec.shape
fig = plt.figure()
librosa.display.specshow(logmelspec, y_axis='mel', x_axis='time', sr=16000)
plt.colorbar()
plt.show()
