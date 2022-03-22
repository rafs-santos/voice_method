import numpy as np
from scipy import io




def load_filter(name):
    fname = io.loadmat('wfk.mat')
    w_name = fname['wf']
    if ("fk4" == name):
        return w_name[0,0][:]
    elif("fk6" == name):
        return w_name[0,1][:]
    elif("fk8" == name):
        return w_name[0,2][:]
    elif("fk14" == name):
        return w_name[0,3][:]
    elif("fk22"):
        return w_name[0,3][:]
    else:
        print("Error filter name")
    return 0

def buffer(x, w_length, hop_size):
    numHops = int(np.floor((x.shape[0]-w_length)/hop_size) + 1)

    if x.ndim > 1:
        r = x.shape[0]
        c = x.shape[1]
    else:
        r = x.shape[0]

    c = 1
    aux_x = np.reshape(x, (r, c))

    y = np.zeros((w_length, numHops*c))

    for channel in range(c):
        for hop in range(numHops):
            temp = aux_x[hop_size*hop:w_length+hop_size*hop, channel]
            #temp = x(1+hopSize*(hop-1):WindowLength+hopSize*(hop-1),channel);
            y[:, hop+channel*numHops] = temp
            #y(:,hop+(channel-1)*numHops) = temp;
    return y

if __name__ == '__main__':
    a = np.arange(1, 11)
    
    print("Windoing array in matrix with hop_size")
    print(a)
    win_length = 5
    hop_length = 2
    b = my_buffer(a, win_length, hop_length)
    print(b)