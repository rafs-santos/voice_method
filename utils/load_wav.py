import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
#import librosa

import os

def loadWave(path, classes):
    dataBase = []
    #z = 0
    for i in range(len(classes)):
        data_aux = []
        for file in os.listdir(path + classes[i]):
            if file.endswith(".wav"):
                data, samplerate = sf.read(os.path.join(path + classes[i], file))
                #data, samplerate = librosa.load(os.path.join(path + classes[i], file)) # very slow
                if (len(data)>=50000):
                    data = data[:50000]
                #print(len(data))
                data_aux.append(data)
        dataBase.append(data_aux)

    return np.array(dataBase, dtype="object"), samplerate

if __name__ == '__main__':
    classes = ["Saud", "Ede", "Nod"]
    path = '../Banco_de_Dados/New_BD/'
    A, B = loadWave(path, classes)
    a = A[0][1]
    print(A.shape)
    plt.plot(a)
    plt.show()
