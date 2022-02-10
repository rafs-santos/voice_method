import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, ifftshift
import pywt

import scipy.io
from aux_algo import load_filter

def removeboundary(wt, VJ, n_length, J, L, scalingvar):
    if (scalingvar):
        wn = np.zeros((J+1, n_length))    
    else:
        wn = np.zeros_like(wt)
    
    LJ = np.zeros((J,), dtype=int)
    MJ = np.zeros((J+1,), dtype=int)
    for jj in range(J):
        LJ[jj] = (2**(jj+1)-1)*(L-1)
        M = int(np.minimum(LJ[jj], n_length))
        wn[jj,:] = wt[jj,:] 
        wn[jj, :M] = np.nan
        MJ[jj] = n_length - M
    
    if(scalingvar):
        wn[jj+1,:] = VJ
        wn[jj+1, :M] = np.nan
        MJ[jj+1] = n_length - M
    else:
        MJ = MJ[:J]
    
    return wn, MJ

def modwtACS(cfs, MJ):
    N = len(cfs)
    cfs = cfs - np.mean(cfs)
    fftpad = int(2**np.ceil(np.log2(2*N)))
    
    wacsDFT = fft(cfs, fftpad)*np.conj(fft(cfs, fftpad))
    wacs = ifftshift(ifft(wacsDFT))
    wacs = 1/MJ *(wacs[fftpad//2:fftpad//2+MJ])

    return wacs

def modwtVAR(cfs, MJ):
    cfs = cfs - np.mean(cfs)
    
    return np.sum(cfs**2)/MJ

def modwtvar(wt, wname):
    # Split in details and aproximation
    VJ = wt[-1,:]
    wt = wt[:-1,:]
    
    # Get legnth of the filter and level of decomposition
    n_level, n_length = wt.shape 
    
    scalingvar = False

    if wname[:2] == "fk":
        w_filter = load_filter(wname)
        L = len(w_filter[0])
    else:
        # Get legnth of the filter
        w_filter = pywt.Wavelet(wname)
        L = len(np.array(w_filter.dec_lo))

    Jmax = int(np.floor(np.log2( (n_length)/(np.max(L-1))+1)))
    if(Jmax < 1):
        print("Error")
        return 0
    Jmax = np.minimum(Jmax, n_level)
    wt = wt[:Jmax,:]
    VJ = VJ[:]

    if(Jmax-n_level == 0):
        scalingvar = True

    # Remove boundary coefficients
    cfs, MJ = removeboundary(wt, VJ, n_length, Jmax, L, scalingvar)

    # Create vector for the variance per scale
    wvar = np.zeros((cfs.shape[0],), dtype=np.float64)
    for jj in range(cfs.shape[0]):
        # Remove Nan from coefficients
        ind_array = ~np.isnan(cfs[jj,:])
        cfsNoNan = cfs[jj,ind_array]
        
        # Get wvar for level jj+1 (taken from the book Walden)
        wvar[jj] = modwtVAR(cfsNoNan, MJ[jj])
        
        # Get wvar for level jj+1 (taken from Matlab)
        #wacs = modwtACS(cfsNoNan, MJ[jj])
        #wvar[jj] = np.real(wacs[0])

    return wvar

def modwtdec(X, G, H, J):
    n_length = len(X)
    upfactor = 2**(J)
    
    Gup = G[np.remainder(upfactor*np.arange(0, n_length), n_length)]
    Hup = H[np.remainder(upfactor*np.arange(0, n_length), n_length)]
    # Updates the new Vhat - coarse coefficients
    Vhat = Gup*X
    # First use Vhat (Notation of book - Walden) old to get the details coefficients
    What = Hup*X

    return Vhat, What

def modwt(x, wname, n_level):
    x = x.ravel()
    if wname[:2] == "fk":
        w_filter = load_filter(wname)
        lo = w_filter[0]
        hi = w_filter[1]
    else:
        # Get the filters to be used
        w_filter = pywt.Wavelet(wname)
        lo = np.array(w_filter.dec_lo, dtype=np.float64)
        hi = np.array(w_filter.dec_hi, dtype=np.float64)

        # MATLAB compatibility -  the filters must be inverted
        lo = np.flipud(lo)
        hi = np.flipud(hi)

    n_length = len(x)

    # Scale the scaling and wavelet filters for the MODWT
    Lo = lo/np.sqrt(2)
    Hi = hi/np.sqrt(2)

    # Allocate coefficient array. Include complexness of x.
    w = np.zeros((n_level+1, n_length), dtype=np.float64)


    # Get the DFT of the filters
    G = fft(Lo, n_length)
    H = fft(Hi, n_length)

    # Get the DFT of the data
    Vhat = fft(x)

    for jj in range(n_level):
        Vhat, What = modwtdec(Vhat, G, H, jj)
        w[jj,:] = np.real(ifft(What))

    w[jj+1,:] = np.real(ifft(Vhat))

    return w

if __name__ == '__main__':
    Fs = 1024
    t = np.arange(0, Fs)*(1/Fs)
    #x = .5*np.sin(2*np.pi*2*t) + .5*np.cos(2*np.pi*t)+.25*np.cos(2*np.pi*10*t)
    #x = x.astype(np.float64)
    mat = scipy.io.loadmat('test.mat')
    mat2 = scipy.io.loadmat('teste.mat')
    
    x = mat['x'].astype(np.float64)
    w = modwt(x, 'db1', 5)

    wvar = modwtvar(w, 'db1')
    print(wvar)
    #plt.show()