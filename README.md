# Voice Method

### This methods is used to analyse voice signal

There are three algorithms the mel spectrogram, mfccs and modwt.
All algorithms using numpy arrays and scipy for fft, ifft, and dct

- To use the modwt algorithm it is necessary install pywavelet that provides the filters used in the method
- To use the mel spectrogram and mfcc is equal to librosa with small changes and must be use librosa because of mel filter bank
