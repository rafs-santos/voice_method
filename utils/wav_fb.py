import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
import pywt


def upSample(h, M):
    h_new = np.zeros(M*len(h))
    h_new[::M] = h
    return h_new

def get_filters(h, g, n_level):
    N = (2**n_level-1)*(len(h)-1)+1
    filters = np.zeros((n_level+1, N))
    filters[0, :len(g)] = g
    if n_level == 1:
        filters[1, :len(h)] = h
        return filters
    aux_g = g
    aux_h = h
    for i in range(1, n_level):
        L = (2**(i+1)-1)*(len(h)-1)+1
        aux_g = upSample(g, 2**i)
        aux_g = np.convolve(aux_g, aux_h)
        filters[i,:L] = aux_g[:L]
        aux_h = np.convolve(upSample(h, 2**i), aux_h)
      
    filters[n_level,:] = aux_h[:L]
  
    return filters


if __name__ == '__main__':
    wname = "db2"
    db = pywt.Wavelet(wname)
    lo = np.array(db.dec_lo)
    hi = np.array(db.dec_hi)

    # Compatilibidade com o matlab os filtros devem ser invertidos
    lo = np.flipud(lo)
    hi = np.flipud(hi)
    
    # Scale the scaling and wavelet filters for the MODWT
    Lo = lo/np.sqrt(2)
    Hi = hi/np.sqrt(2)

    a = get_filters(Lo, Hi, 6)
    n_fft = 4096
    H_r = fft(a,n_fft)
    freq = np.arange(0, n_fft//2+1)
    aux_f = np.tile(freq, (12,1))
    print(aux_f.shape)
    print(H_r.shape)
    # Plot figure
    fig, ax = plt.subplots(figsize=(16,9))
    for i in range(H_r.shape[0]):
      ax.plot(freq, np.abs(H_r[i,:n_fft//2+1]), 'b')
      ax.axis('off')
    path = "../../../Latex/Defesa/Dissertacao/Capitulo3/Figuras/"
    name_fig = "wavFB"

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    #plt.savefig(path+name_fig+".pdf",bbox_inches='tight', pad_inches = 0.05)

    plt.show()

    plt.show()
