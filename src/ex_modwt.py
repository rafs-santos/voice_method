import sys, os
sys.path.insert(0, './utils')

import matplotlib.pyplot as plt
import numpy as np
import librosa

from algo_modwt import modwt, modwtvar

from play_audio import playsound

if __name__ == '__main__':
    wname =  'db2'
    y, sr = librosa.load(librosa.ex('trumpet'))
    n_level = 5


    w = modwt(y, wname, n_level)
    wvar = modwtvar(w, wname)

    fig, ax = plt.subplots(w.shape[0] + 1, 1, figsize=(16, 8), sharex=True)
    ax[0].plot(y)
    ax[0].set(title='Signal')
    for i in range(w.shape[0]-1):
        ax[i+1].plot(w[i])
        ax[i+1].set(ylabel='D'+str(i+1))
    
    ax[-1].plot(w[w.shape[0]-1])
    ax[-1].set(ylabel='A'+str(w.shape[0]-1))
    print(wvar)

    data = w[-1] 
    data = data / np.max(np.abs(data))
    playsound(data,sr)

    plt.show()