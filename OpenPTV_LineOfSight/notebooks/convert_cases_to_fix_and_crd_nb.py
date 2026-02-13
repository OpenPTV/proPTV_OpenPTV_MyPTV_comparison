import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    # This notebook is to convert calibration cases created by Robin into the format that can be used by the calibration tool
    # The calibration tool requires the following format:
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import os

    return np, os


@app.cell
def _():
    # OpenPTV requires for the case of Multiplane calibration the following format:
    # .crd and .fix files


    # plane1_cam1.tif.fix:
    """ 
       0.00000   15.00000   15.00000  274.00000
       1.00000   30.00000   15.00000  274.00000
       2.00000   45.00000   15.00000  274.00000
       3.00000   60.00000   15.00000  274.00000
       4.00000   75.00000   15.00000  274.00000
       5.00000   90.00000   15.00000  274.00000
    """

    # plane1_cam1.tif.crd: 
    """
      0.00000 315.94791 1997.48086
      1.00000 411.23412 1997.85856
      2.00000 507.31137 1998.25029
      3.00000 604.33361 1998.40124
      4.00000 702.21405 1998.50122
      5.00000 800.80803 1998.78058
      6.00000 900.03065 1998.74938
      7.00000 1000.25174 1998.77980
    """

    # /cases folder contain the data in Robin's format:
    # markers_c0.txt

    """
    # x,y,X,Y,Z
    3.150000000000000000e+02 1.996000000000000000e+03 1.500000000000000000e+01 1.500000000000000000e+01 2.740000000000000000e+02
    4.100000000000000000e+02 1.997000000000000000e+03 3.000000000000000000e+01 1.500000000000000000e+01 2.740000000000000000e+02
    5.060000000000000000e+02 1.997000000000000000e+03 4.500000000000000000e+01 1.500000000000000000e+01 2.740000000000000000e+02
    6.030000000000000000e+02 1.997000000000000000e+03 6.000000000000000000e+01 1.500000000000000000e+01 2.740000000000000000e+02
    7.010000000000000000e+02 1.998000000000000000e+03 7.500000000000000000e+01 1.500000000000000000e+01 2.740000000000000000e+02
    8.000000000000000000e+02 1.998000000000000000e+03 9.000000000000000000e+01 1.500000000000000000e+01 2.740000000000000000e+02
    """
    return


@app.cell
def _(np):
    class Parameter:
        cams = [0,1,2,3]
        Vmin = [0,0,0]
        Vmax = [300,300,300]
        N1, N2 = 361, 5

    params = Parameter()

    markers = [np.loadtxt('../cases/case_allmarkers/markers_c'+str(cam)+'.txt') for cam in params.cams]
    return markers, params


@app.cell
def _(markers, np, os, params):
    """
    Saves detected and known calibration points in crd and fix format, respectively.
    These files are needed for multiplane calibration.
    """
    path = '/home/user/Downloads/rbc300/cal'
    for c in params.cams:

        XYZ = markers[c][:,2:]
        xy = markers[c][:,:2]

        for plane, z in enumerate(np.unique(XYZ[:,2])):
            txt_detected = os.path.join(path, 'plane'+str(plane+1)+'_cam'+str(c+1)+'.tif.crd')
            txt_matched = os.path.join(path, 'plane'+str(plane+1)+'_cam'+str(c+1)+'.tif.fix')

            ind = np.argwhere(XYZ[:,2]==z)[:,0]
            detected = np.c_[np.arange(len(ind)), xy[ind]]
            known = np.c_[np.arange(len(ind)), XYZ[ind]]

            np.savetxt(txt_detected, detected, fmt="%9.5f")
            np.savetxt(txt_matched, known, fmt="%10.5f")
    return


if __name__ == "__main__":
    app.run()
