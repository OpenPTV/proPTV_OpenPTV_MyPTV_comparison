import numpy as np

def LoadMarkerList(cam):
    return np.loadtxt('markers_c'+str(cam)+'.txt')

def SaveMarkerList(data):
    # Format: x0, y0, x1, y1, x2, y2, x3, y3, X, Y, Z, dX, dY, dZ
    return np.savetxt('markers_error.txt',data)