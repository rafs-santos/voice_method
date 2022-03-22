import sys, os
sys.path.insert(0, './utils')

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from algo_mel import melspec
from play_audio import playsound

if __name__ == '__main__':
    y, sr = librosa.load(librosa.ex('trumpet'))

    #y = y / np.max(np.abs(y))
    #playsound(y,sr)

    # compare with librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000, center=False)
    # method
    S1 = melspec(data=y, fs=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000)

    print(S.shape)
    print(S1.shape)
    for S in [S, S1]:
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')

    plt.show()
    
