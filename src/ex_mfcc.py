import sys, os
sys.path.insert(0, './utils')

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from algo_mel import melspec, mfcc
from play_audio import playsound

if __name__ == '__main__':
    y, sr = librosa.load(librosa.ex('libri1'))

    #y = y / np.max(np.abs(y))
    #playsound(y,sr)

    # compare with librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000, center=False)
    # method
    S1 = melspec(data=y, fs=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000)

    mfcc01 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, center=False)

    mfcc02 = mfcc(data=y, fs=sr, n_mfcc=40, win_length=2048, hop_length=512)

    print(mfcc01.shape)
    print(mfcc02.shape)

    melS = [S, S1]
    mfcc = [mfcc01, mfcc02]
    for i in range(2):
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img = librosa.display.specshow(librosa.power_to_db(melS[i], ref=np.max),
                                x_axis='time', y_axis='mel', fmax=8000,
                                ax=ax[0])
        fig.colorbar(img, ax=[ax[0]])
        ax[0].set(title='Mel spectrogram')
        ax[0].label_outer()
        img = librosa.display.specshow(mfcc[i], x_axis='time', ax=ax[1])
        fig.colorbar(img, ax=[ax[1]])
        ax[1].set(title='MFCC')
    
    plt.show()
    
