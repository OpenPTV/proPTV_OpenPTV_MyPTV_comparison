import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From proPTV to OpenPTV

    Let us imagine the following thought experiment. Robin has performed a physical calibration using Soloff method. The only physical information available is the size of the image, several vectors of Soloff coefficients and particle images. Of course, proPTV is sufficient to convert those to the 3D positions and track them in time.

    What if another software can do something in addtion, e.g. track in 3D-2D space? Option 1 is to recode all the knowledge again for the proPTV. Option 2 is to create OpenPTV working folder with all the missing information, calibrate OpenPTV with the virtual scene created using proPTV and then use openPTV capabilities. For instance, OpenPTV has dumbbell or wand calibration, and MyPTV has an extended Soloff method.

    OpenPTV can be calibrated in one of the two ways, depending on the available or reliable assumption of the physical setup: a) using single homogeneous media approximation, ``all-in-air'' with $n_1 = n_2 = n_3 = 1$ and then it would be roughly the simple Tsai model with aberrations; b) use multi-media model $n_1 = 1, n_2 = 1.46, n_3 = 1.33$ and then we need additional physical information: glass thickness, water distance, parallel positioning of the target in respect to the glass window and so on.
    """)
    return


@app.cell
def _():
    # Implementation
    # Create a 3D target, a list of 3D points well distributed across the 3d volume
    # Create 2D projections of the 3D target on the cameras using proPTV 
    # use 2D projections per cameras and 3D target positions along with the assumed parameters
    # for the OpenPTV to create new calibration parameters
    # compare the results using the 3D Line of Sight
    return


@app.cell
def _():
    # %load proPTV_to_OpenPTV.py
    # %% [markdown]
    # # From proPTV to OpenPTV
    # 
    # Let us imagine the following thought experiment. Robin has performed a physical calibration using Soloff method. The only physical information available is the size of the image, several vectors of Soloff coefficients and particle images. Of course, proPTV is sufficient to convert those to the 3D positions and track them in time. 
    # 
    # What if another software can do something in addtion, e.g. track in 3D-2D space? Option 1 is to recode all the knowledge again for the proPTV. Option 2 is to create OpenPTV working folder with all the missing information, calibrate OpenPTV with the virtual scene created using proPTV and then use openPTV capabilities. For instance, OpenPTV has dumbbell or wand calibration, and MyPTV has an extended Soloff method. 
    # 
    # OpenPTV can be calibrated in one of the two ways, depending on the available or reliable assumption of the physical setup: a) using single homogeneous media approximation, ``all-in-air'' with $n_1 = n_2 = n_3 = 1$ and then it would be roughly the simple Tsai model with aberrations; b) use multi-media model $n_1 = 1, n_2 = 1.46, n_3 = 1.33$ and then we need additional physical information: glass thickness, water distance, parallel positioning of the target in respect to the glass window and so on. 
    # 

    # %%
    # Implementation
    # Create a 3D target, a list of 3D points well distributed across the 3d volume
    # Create 2D projections of the 3D target on the cameras using proPTV 
    # use 2D projections per cameras and 3D target positions along with the assumed parameters
    # for the OpenPTV to create new calibration parameters
    # compare the results using the 3D Line of Sight


    # %%
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # '%matplotlib widget' command supported automatically in marimo


    # Create a linear space from 0 to 300 with 5 points in each direction
    x = np.linspace(10, 290, 5)
    y = np.linspace(10, 290, 5)
    z = np.linspace(10, 290, 5)

    # Create a 3D grid of points
    X, Y, Z = np.meshgrid(x, y, z)


    # %%
    from openptv_python.tracking_frame_buf import TargetArray
    from openptv_python.calibration import Calibration
    from openptv_python.parameters import ControlPar, VolumePar, OrientPar, MultimediaPar
    from openptv_python.orientation import full_calibration, external_calibration
    from openptv_python.trafo import metric_to_pixel
    from openptv_python.imgcoord import img_coord

    return (
        Calibration,
        ControlPar,
        OrientPar,
        TargetArray,
        VolumePar,
        X,
        Y,
        Z,
        external_calibration,
        full_calibration,
        img_coord,
        metric_to_pixel,
        np,
        plt,
    )


@app.cell
def _(ControlPar, OrientPar, VolumePar):
    import pathlib, os
    working_path = pathlib.Path('OpenPTV_LineOfSight')

    control_parameters = ControlPar().from_file(os.path.join(working_path,"parameters","ptv.par"))
    volume_parameters = VolumePar().from_file(os.path.join(working_path,"parameters","criteria.par"))
    orient_parameters = OrientPar()
    return control_parameters, orient_parameters, os, working_path


@app.cell
def _(img_coord, metric_to_pixel, np):
    # orient_parameters.scxflag = 1 # updated in the future version
    # control_parameters.mm = MultimediaPar(nlay=1,n1=1., n2=[1.], d=[1.],n3=1.)

    # %%

    def openptv_project_XYZ_on_camera(XYZ, cal, cpar):
        """ Projects 3D points vector XYZ on camera defined by cal, cpar """
        xy = np.empty((XYZ.shape[0],2))
        for i, point in enumerate(XYZ):
            x, y = img_coord(point, cal, cpar.mm)
            x, y = metric_to_pixel(x, y, cpar)
            xy[i,0] = x
            xy[i,1] = y
        return xy

    return (openptv_project_XYZ_on_camera,)


@app.cell
def _(X, Y, Z, plt):
    # %%
    # Bring in proPTV data and functions
    import sys
    sys.path.append('proPTV_LineOfSight')
    from func import F
    calibration_path = 'proPTV_LineOfSight/calibration/c{cam}/soloff_c{cam}{xy}.txt'
    Vmin, Vmax = ([0, 0, 0], [300, 300, 300])
    _axis = plt.figure(figsize=(5, 5)).add_subplot(projection='3d')
    (_axis.set_xlabel('X'), _axis.set_ylabel('Y'), _axis.set_zlabel('Z'))
    (_axis.plot([0, 300], [0, 0], [0, 0], c='black'), _axis.plot([0, 0], [0, 300], [0, 0], c='black'), _axis.plot([0, 0], [0, 0], [0, 300], c='black'))
    (_axis.plot([300, 300], [0, 300], [0, 0], c='black'), _axis.plot([300, 300], [0, 0], [0, 300], c='black'), _axis.plot([0, 300], [300, 300], [0, 0], c='black'))
    # make 3D figure
    (_axis.plot([0, 0], [300, 300], [0, 300], c='black'), _axis.plot([0, 300], [0, 0], [300, 300], c='black'), _axis.plot([0, 0], [0, 300], [300, 300], c='black'))
    (_axis.plot([300, 300], [300, 300], [0, 300], c='black'), _axis.plot([300, 300], [0, 300], [300, 300], c='black'), _axis.plot([0, 300], [300, 300], [300, 300], c='black'))
    # plot box
    _axis.scatter(X, Y, Z, 'o', color='b')
    plt.tight_layout()
    # plot torus
    # axis.plot(X_mid,Y_mid,Z_mid,'o-', color='red', linewidth=1)
    # axis.plot(X_spiral,Y_spiral,Z_spiral,'o-', color='green', linewidth=1)
    plt.show()
    return F, calibration_path


@app.cell
def _(
    Calibration,
    F,
    TargetArray,
    X,
    Y,
    Z,
    calibration_path,
    control_parameters,
    external_calibration,
    full_calibration,
    np,
    openptv_project_XYZ_on_camera,
    orient_parameters,
    os,
    plt,
    working_path,
):
    # make projection of lines in cam c
    fig, _axis = plt.subplots(2, 2, figsize=(8, 8))
    for c in range(4):
        ax, ay = (np.loadtxt(calibration_path.format(cam=c, xy='x'), delimiter=','), np.loadtxt(calibration_path.format(cam=c, xy='y'), delimiter=','))
        xy = np.vstack([F(np.vstack([X.flat, Y.flat, Z.flat]).T, ax), F(np.vstack([X.flat, Y.flat, Z.flat]).T, ay)]).T
        _axis[int(c / 2), c % 2].set_title('reprojection on camera ' + str(c))
        _axis[int(c / 2), c % 2].imshow(np.zeros([2160, 2560]), cmap='gray')  # load calibration of camera c
        _axis[int(c / 2), c % 2].plot(xy[:, 0], xy[:, 1], '+', c='white')
        openptv_camera_calibration = Calibration().from_file(os.path.join(working_path, 'calibration', f'cam{c + 1}.tif.ori'), None)
        all_known = np.c_[X.flat, Y.flat, Z.flat]
        all_detected = xy.copy()  # Comment out when you have all calibration parameters
        targs = TargetArray(len(all_detected))  # ax, ay = np.loadtxt(calibration_path.format(cam=c,xy="x"),delimiter=','), np.loadtxt(calibration_path.format(cam=c,xy="y"),delimiter=',')
        for tix in range(len(all_detected)):
            targ = targs[tix]  # estimate projection of the lines in camera c
            det = all_detected[tix]
            targ.set_pnr(tix)
            targ.set_pos(det)
        print(openptv_camera_calibration.ext_par, openptv_camera_calibration.int_par)
        openptv_camera_calibration = Calibration()
        openptv_camera_calibration.ext_par.x0 = 0
        openptv_camera_calibration.ext_par.y0 = 0
        openptv_camera_calibration.ext_par.z0 = 1000.0
        openptv_camera_calibration.int_par.cc = 41.0
        openptv_camera_calibration.glass_par = np.array([0.0, 0.0, 300.0])
        print(f' Before: openptv_camera_calibration.ext_par ={openptv_camera_calibration.ext_par!r}, openptv_camera_calibration.int_par = {openptv_camera_calibration.int_par!r}')
        outcome = external_calibration(openptv_camera_calibration, all_known, all_detected, control_parameters)
        assert outcome is True
        print(f' After raw calibration: openptv_camera_calibration.ext_par ={openptv_camera_calibration.ext_par!r}, openptv_camera_calibration.int_par = {openptv_camera_calibration.int_par!r}')
        xy_optv = openptv_project_XYZ_on_camera(np.vstack([X.flat, Y.flat, Z.flat]).T, openptv_camera_calibration, control_parameters)
        _axis[int(c / 2), c % 2].plot(xy_optv[:, 0], xy_optv[:, 1], 'x', c='red')
        residuals, targ_ix, err_est = full_calibration(openptv_camera_calibration, all_known, targs, control_parameters, orient_parameters, dm=0.0001, drad=0.0001)
        print(f' After fine calibration 1: openptv_camera_calibration.ext_par ={openptv_camera_calibration.ext_par!r}, openptv_camera_calibration.int_par = {openptv_camera_calibration.int_par!r}')
        xy_optv = openptv_project_XYZ_on_camera(np.vstack([X.flat, Y.flat, Z.flat]).T, openptv_camera_calibration, control_parameters)
        _axis[int(c / 2), c % 2].plot(xy_optv[:, 0], xy_optv[:, 1], '+', c='magenta')
        orient_parameters.ccflag = 0
        orient_parameters.xhflag = 0
        orient_parameters.yhflag = 0
        orient_parameters.scxflag = 1
        residuals, targ_ix, err_est = full_calibration(openptv_camera_calibration, all_known, targs, control_parameters, orient_parameters, dm=1e-11, drad=1e-11)  # %%
        xy_optv = openptv_project_XYZ_on_camera(np.vstack([X.flat, Y.flat, Z.flat]).T, openptv_camera_calibration, control_parameters)
    # fig.show()
        _axis[int(c / 2), c % 2].plot(xy_optv[:, 0], xy_optv[:, 1], '.', c='magenta')  # now we try to reset calibration and work from zero setup:  # good initial guess  # nobody can guess  # since we use multi-media, we better set the in-water distance  # using glass vector  # 300 mm tank width  # start with the simplest case: all in air:  # control_parameters.mm = MultimediaPar(nlay=1,n1=1., n2=[1.], d=[1.],n3=1.)  # Run the multiplane calibration in full  # let some parameters be fixed to change:
    return


if __name__ == "__main__":
    app.run()
