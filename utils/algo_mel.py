import librosa
import numpy as np

from scipy import signal
from scipy.fftpack import dct
from scipy.fft import fft

from aux_algo import buffer


def melspec(data, fs=22050, n_fft=2048, hop_length=512, win_length=None, window='hann', power=2.0, htk=False, n_mels=128, fmin=0, fmax=None):
    """
    mel spectrogram using librosa library

    # Parameters
    
    data : numpy array
        audio time-series

    fs : number > 0 
        sample rate

    n_fft : int > 0
        number of points in fft >= win_length
    
    hop_length : int > 0
        number of samples between successive frames
    
    win_length : int <= n_fft
        windowed frames 
    
    window : string
        get window with `scipy.signal.get_window`
    
    power : float > 0
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power
    
    fmin : float >= 0 
        lowest frequency (in Hz)

    fmax : float >= 0 
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    
    More information see ``librosa.mel``

    #Returns
    
    M : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    """

    if fmax == None:
        fmax = fs / 2.0

    if win_length == None:
        win_length = n_fft
    
    melFB = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk)

    mt_data = buffer(data, win_length, hop_length)
    wind = signal.get_window(window, win_length) 
    
    mt_data = mt_data*wind.reshape((win_length,1))
    stft = fft(mt_data, n_fft, axis=0)[:n_fft//2+1]

    stft = np.abs(stft)**power

    return np.dot(melFB, stft)

def mfcc(data=None, fs=22050, n_mfcc=20, dct_type=2, norm="ortho", LogEnergy=None, win_length=2048, hop_length=512, **kwargs):
    """
    Read mel spectrogram and librosa
    """
    if LogEnergy == 'Append' or LogEnergy == 'Replace':
        mt_data = buffer(data, win_length, hop_length)
        energyLog = np.sum(mt_data**2, axis=0)

    melSpec = melspec(data=data, fs=fs, win_length=win_length, hop_length= hop_length, **kwargs)

    melSpec = librosa.power_to_db(melSpec)

    mfcc = dct(melSpec, axis=0, type=dct_type, norm=norm)[:n_mfcc]

    if LogEnergy == 'Append':
        mfcc = np.concatenate((energyLog, mfcc[0:,:]), axis=0)
    elif LogEnergy == 'Replace':
        mfcc = np.concatenate((energyLog, mfcc[1:,:]), axis=0)
    
    return mfcc
