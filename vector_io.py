import numpy as np
import struct



def ivecs_read(fname: str) -> np.ndarray:
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]
    # .copy()
 
 
def fvecs_read(fname: str) -> np.ndarray:
    return ivecs_read(fname).view('float32')



def fvecs_write(fname: str, data) -> None:
    d = struct.pack('I', data.shape[1]) # y.size will be writed in uint format
    with open(fname, 'wb') as fp:
        for vec in data:
            fp.write(d)
            for x in vec:
                fp.write(struct.pack('f', x))

def ivecs_write(fname: str, data) -> None:
    d = struct.pack('I', data.shape[1]) # y.size will be writed in uint format
    with open(fname, 'wb') as fp:
        for vec in data:
            fp.write(d)
            for x in vec:
                fp.write(struct.pack('I', x))