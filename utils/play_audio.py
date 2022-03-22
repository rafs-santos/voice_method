import sounddevice as sd
import threading
import numpy as np
import matplotlib.pyplot as plt

import load_wav

def play(sound, samplerate, k):
    event = threading.Event()
    def callback(outdata, frames, time, status):
        nonlocal k
        if ((k+1)*frames >= len(sound)):
            data = sound[k*frames:]
            raise sd.CallbackStop
        else:
            data = sound[k*frames:(k+1)*frames]

        if(len(outdata)>len(data)):
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        else:
            outdata[:] = data
        k = k+1
    #channels = sound.shape
    channels = sound.ndim
    chunk = 8192
    sound = sound.reshape((len(sound), channels))
    with sd.OutputStream(samplerate = samplerate,
                                channels = channels,
                                callback = callback,
                                blocksize = chunk,
                                finished_callback = event.set) as stream:

        with stream:
            event.wait()


def playsound(data, fs):
    """
    Scale the sound to lim [-1, 1] based on soundsc matlab
    """
    xmax = np.amax(np.absolute(data))
    slim = np.array([-xmax, xmax])
    dx = np.diff(slim)
    if dx==0:
        data = np.zeros_like(data)
    else:
        data = (data-slim[0])/dx*2-1
    """
    Play sound
    """
    new_thread = threading.Thread(target=play, args=(data, fs, 0))
    new_thread.start()



if __name__ == '__main__':
    classes = ["Saud", "Ede", "Nod"]
    path = '../Banco_de_Dados/New_BD/'
    data, fs = load_wav.loadWave(path, classes)

 
    n_Class = len(classes)
    n_Sig = len(data[0])
    y = data[2][7]
    y = y[:26518]
    y = y / np.max(np.abs(y))
    playsound(y,fs)
    """ 
    fs = 44100
    t = np.linspace(0, 3, 3*fs)
    
    y = np.cos(2*np.pi*5000*t)
    playsound(y,fs)
    print("acabou")
    """

#print(y.shape)

"""
######################### Função que utiliza passos para saber o último termo da PA
"""
"""def limSig(data, L, overlap):
    length_data = len(data)
    k = True
    n = 1
    
    r  = L-int(overlap*L)
    while k:
        aux = L+(n-1)*r
        if(aux > length_data):
            aux = L+(n-2)*r
            k = False
            
        n = n + 1

    return data[:aux]"""



#print(type(block))
'''def _play(sound, samplerate):
    event =threading.Event()

    def callback(outdata, frames, time, status):
        data = wf.buffer_read(frames, dtype='float32')
        if len(outdata) > len(data):
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
            raise sd.CallbackStop
        else:
            outdata[:] = data

    with sf.SoundFile(sound) as wf:
        stream = sd.RawOutputStream(samplerate=wf.samplerate,
                                    channels=wf.channels,
                                    callback=callback,
                                    blocksize=1024,
                                    finished_callback=event.set)
        with stream:
            event.wait()

def _playsound(sound, fs):
    new_thread = threading.Thread(target=_play, args=(sound,fs))
    new_thread.start()
'''



'''
import matplotlib.pyplot as plt

import soundfile as sf
import sounddevice as sd
import threading
import numpy as np

def loadWav(wavFile):
    data, samplerate = sf.read(wavFile)
    return data, samplerate

x, fs = loadWav('sound.wav')

plt.plot(x)
plt.show()
'''
'''
# Load the whole file in memory to play
def playAudio(data, fs):
    sd.play(data, fs)
    sd.wait() 


fs = 44100
t = np.linspace(0, 3, 3*fs)

y = np.cos(2*np.pi*5000*t)

playAudio(y,fs)

'''
