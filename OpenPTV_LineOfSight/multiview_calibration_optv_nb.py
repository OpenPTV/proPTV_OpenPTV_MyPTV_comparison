import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import sys
    import cv2
    import numpy as np
    import matplotlib
    # Set the matplotlib backend before importing pyplot
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    from scipy import linalg

    return (np,)


@app.cell
def _():
    # Import from the new implementation in multiview_calibration.py
    from multiview_calibration import DLT, Parameter, main

    return (main,)


@app.cell
def _(main):
    # Run main()
    main()
    return


@app.cell
def _():
    # Copy of the old code for reference



    # # convention marker sort: left to right (x) then top to bottom (y)
    # # note that y top denotes to the smallest y because in the image the y axis is inversed+
    # # Therefore, in the images the top left point is (0,0,0) in the first frame

    # # parameter
    # N_t = np.arange(1,31) # manuel input the number of frames per camera
    # marker_distance, Z0 = (120, 120), 0 # X[mm], Y[mm], Z0[mm] , shift -360 in x and +360 in y for davis
    # marker_size = (8, 7)  # x, y
    # image_size = (2560, 2048)  # x[px], y[px]

    # # define first plane XYZ
    # X, Y, Z = np.meshgrid(np.arange(0,marker_size[0]*marker_distance[0],marker_distance[0]),-np.arange(0,marker_size[1]*marker_distance[1],marker_distance[1]),np.linspace(Z0,Z0,1))
    # XYZ = np.asarray(np.vstack([X.ravel(),Y.ravel(),Z.ravel()]).T, dtype=np.float32)

    # # calibrate camera 0
    # XYZ_0 = [XYZ for t in N_t]
    # xy_0 = [np.asarray(np.loadtxt('markers_xy/c{cam}/c{cam}_{time}.txt'.format(cam=0,time=str(t).zfill(5)),skiprows=1), dtype=np.float32) for t in N_t]
    # ret_0, M_0, d_0, r_0, t_0 = cv2.calibrateCamera(XYZ_0,xy_0,image_size,None,None) 
    # R_0 = cv2.Rodrigues(r_0[0])[0]    
    # pos_c0 = -np.dot(R_0.T, t_0[0]).ravel()
    # print('position cam 0: ', pos_c0)
    # # calibrate camera 1
    # XYZ_1 = [XYZ for t in N_t]
    # xy_1 = [np.asarray(np.loadtxt('markers_xy/c{cam}/c{cam}_{time}.txt'.format(cam=1,time=str(t).zfill(5)),skiprows=1), dtype=np.float32) for t in N_t]
    # ret_1, M_1, d_1, r_1, t_1 = cv2.calibrateCamera(XYZ_1,xy_1,image_size,None,None) 
    # R_1 = cv2.Rodrigues(r_1[0])[0]    
    # pos_c1 = -np.dot(R_1.T, t_1[0]).ravel()
    # print('position cam 1: ', pos_c1)
    # # calibrate camera 2
    # XYZ_2 = [XYZ for t in N_t]
    # xy_2 = [np.asarray(np.loadtxt('markers_xy/c{cam}/c{cam}_{time}.txt'.format(cam=2,time=str(t).zfill(5)),skiprows=1), dtype=np.float32) for t in N_t]
    # ret_2, M_2, d_2, r_2, t_2 = cv2.calibrateCamera(XYZ_2,xy_2,image_size,None,None) 
    # R_2 = cv2.Rodrigues(r_2[0])[0]    
    # pos_c2 = -np.dot(R_2.T, t_2[0]).ravel()
    # print('position cam 2: ', pos_c2)
    # # calibrate camera 3
    # XYZ_3 = [XYZ for t in N_t]
    # xy_3 = [np.asarray(np.loadtxt('markers_xy/c{cam}/c{cam}_{time}.txt'.format(cam=3,time=str(t).zfill(5)),skiprows=1), dtype=np.float32) for t in N_t]
    # ret_3, M_3, d_3, r_3, t_3 = cv2.calibrateCamera(XYZ_3,xy_3,image_size,None,None) 
    # R_3 = cv2.Rodrigues(r_3[0])[0]    
    # pos_c3 = -np.dot(R_3.T, t_3[0]).ravel()
    # print('position cam 3: ', pos_c3)

    # # stereo matching of the marker positons - use only cam 0 and cam 1
    # ret, CM0, dist0, CM1, dist1, R, T, E, F = cv2.stereoCalibrate(XYZ_0[0:1], xy_0[0:1], xy_1[0:1], M_0, d_0, M_1, d_1, image_size)
    # #projection matrix for camera 0
    # RT0 = np.concatenate([R_0, t_0[0]], axis = -1)
    # P0 = M_0 @ RT0 
    # #projection matrix for camera 1
    # RT1 = np.concatenate([R@R_0, (R@t_0[0]+T)], axis = -1)
    # P1 = M_1 @ RT1 
    # # P contains the 3D marker positions of each plate at each position
    # P = []
    # for i in range(len(N_t)):
    #     xyz = []
    #     for xy0, xy1 in zip(xy_0[i],xy_1[i]):
    #         xyz.append(DLT(P0, P1, xy0, xy1))
    #     P.append(np.asarray(xyz,dtype=np.float32))

    # # recalibrate camera 0
    # XYZ_0 = [np.asarray(Pj,dtype=np.float32) for Pj in P]
    # xy_0 = [np.asarray(np.loadtxt('markers_xy/c{cam}/c{cam}_{time}.txt'.format(cam=0,time=str(t).zfill(5)),skiprows=1), dtype=np.float32) for t in N_t]
    # ret_0, M_0, d_0, r_0, t_0 = cv2.calibrateCamera(XYZ_0,xy_0,image_size,M_0,d_0,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    # R_0 = cv2.Rodrigues(r_0[0])[0]    
    # pos_c0_0 = -np.dot(R_0.T, t_0[0]).ravel()
    # print('new position cam 0: ', pos_c0_0)
    return


@app.cell
def _():
    ## Old code for recalibration of all cameras - not used anymore
    ## not clear why we needed to recalibrate: TODO: check if necessary


    # # recalibrate camera 1
    # XYZ_1 = [np.asarray(Pj,dtype=np.float32) for Pj in P]
    # xy_1 = [np.asarray(np.loadtxt('markers_xy/c{cam}/c{cam}_{time}.txt'.format(cam=1,time=str(t).zfill(5)),skiprows=1), dtype=np.float32) for t in N_t]
    # ret_1, M_1, d_1, r_1, t_1 = cv2.calibrateCamera(XYZ_1,xy_1,image_size,M_1,d_1,flags=cv2.CALIB_USE_INTRINSIC_GUESS) 
    # # recalibrate camera 2
    # XYZ_2 = [np.asarray(Pj,dtype=np.float32) for Pj in P]
    # xy_2 = [np.asarray(np.loadtxt('markers_xy/c{cam}/c{cam}_{time}.txt'.format(cam=2,time=str(t).zfill(5)),skiprows=1), dtype=np.float32) for t in N_t]
    # ret_2, M_2, d_2, r_2, t_2 = cv2.calibrateCamera(XYZ_2,xy_2,image_size,M_2,d_2,flags=cv2.CALIB_USE_INTRINSIC_GUESS) 
    # # recalibrate camera 3
    # XYZ_3 = [np.asarray(Pj,dtype=np.float32) for Pj in P]
    # xy_3 = [np.asarray(np.loadtxt('markers_xy/c{cam}/c{cam}_{time}.txt'.format(cam=3,time=str(t).zfill(5)),skiprows=1), dtype=np.float32) for t in N_t]
    # ret_3, M_3, d_3, r_3, t_3 = cv2.calibrateCamera(XYZ_3,xy_3,image_size,M_3,d_3,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    return


@app.cell
def _():
    # import matplotlib.pyplot as plt
    from pathlib import Path
    from optv.parameters import ControlParams, VolumeParams
    from optv.calibration import Calibration
    from optv.orientation import external_calibration
    from optv.imgcoord import image_coordinates
    from optv.transforms import convert_arr_metric_to_pixel, convert_arr_pixel_to_metric, distorted_to_flat
    from optv.orientation import point_positions
    from optv.correspondences import MatchedCoords
    from optv.tracking_framebuf import TargetArray
    from scipy.optimize import minimize
    import pandas as pd
    # import plotly.express as px
    # import plotly.figure_factory as ff
    # import plotly.graph_objects as go
    import plotly.express as px

    return (
        Calibration,
        MatchedCoords,
        TargetArray,
        convert_arr_metric_to_pixel,
        image_coordinates,
        minimize,
        point_positions,
    )


@app.cell
def _(
    Calibration,
    MatchedCoords,
    TargetArray,
    convert_arr_metric_to_pixel,
    image_coordinates,
    np,
    point_positions,
):
    def array_to_calibration(x:np.ndarray, cal:Calibration) -> None:
        cal.set_pos(x[:3])
        cal.set_angles(x[3:6])
        cal.set_primary_point(x[6:9])
        cal.set_radial_distortion(x[9:12])
        cal.set_decentering(x[12:14])
        cal.set_affine_trans(x[14:])
        return None

    def calibration_to_array(cal:Calibration) -> np.ndarray:
        return np.concatenate([
            cal.get_pos(),
            cal.get_angles(),
            cal.get_primary_point(),
            cal.get_radial_distortion(),
            cal.get_decentering(),
            cal.get_affine(),
        ])

    def error_function(x, cal, XYZ, xy, cpar):
    
        array_to_calibration(x, cal)

        targets = convert_arr_metric_to_pixel(
            image_coordinates(XYZ, cal, cpar.get_multimedia_params()),
        cpar,
        )
        # err = np.sum(np.abs(xy - targets))
        err = np.sum((xy - targets)**2)
        # print(err)
        return err

    def targetize(detects, approx_size, sumg=10):
        """
        Creates a correct TargetArray object with the detected positions and some
        placeholder values for target parameters that I don't use.
    
        Arguments:
        detects - (n,2) array, pixel coordinates of a detected target.
        approx_size - a value to use for the pixel size placeholders.
        sumg - a value to use for the sum of grey values placeholder.
            Default: 10.
        """
        targs = TargetArray(len(detects))
    
        tnum = 0
        for t, pos in zip(targs, detects):
            t.set_pos(pos)
            t.set_pnr(tnum)
            t.set_sum_grey_value(sumg) # whatever
            t.set_pixel_counts(approx_size**2 * 4, approx_size*2, approx_size*2)
            t.set_tnr(-1) # The official "correspondence not found" that 
                                   # the rest of the code expects.
            tnum += 1
    
        return targs

    def pixel_to_3d(markers, cpar, cals, vpar):
        """ converts numpy array of size (2,) from pixel to flat coordinates"""
        detected = []
        corrected = []
        pnrs = []
        for cix in range(cpar.get_num_cams()):
            targs = targetize(markers[cix][:,:2], 1,1)
            # targs.sort_y()  # not sure why it matters but it does
        
            detected.append(targs)
            pnrs.append([t.pnr() for t in targs])

            # mc = 
            # _, pnr = mc.as_arrays()
            # pnrs.append(pnr)
            corrected.append(MatchedCoords(targs, cpar, cals[cix]))

        flat = np.array([corrected[cix].get_by_pnrs(np.array(pnrs[cix])) \
                for cix in range(len(cals))])

        pos3d, rcm = point_positions(flat.transpose(1,0,2), cpar, cals, vpar)

        return pos3d, rcm

    return array_to_calibration, calibration_to_array, error_function


@app.cell
def _():
    # Meanwhile pyPTV changed the parameter format to YAML:
    from pyptv.parameter_manager import ParameterManager
    from pyptv.ptv import _populate_cpar, _populate_spar, _populate_vpar, _read_calibrations


    pm = ParameterManager()
    pm.from_yaml('parameters_Run1.yaml')


    # shortcuts
    params = pm.parameters
    num_cams = pm.num_cams

    cpar = _populate_cpar(params['ptv'], num_cams)
    spar = _populate_spar(params['sequence'], num_cams)
    vpar = _populate_vpar(params['criteria'])

    cals = _read_calibrations(cpar, num_cams)
    return cals, cpar


@app.cell
def _(
    XYZ_0,
    XYZ_1,
    XYZ_2,
    XYZ_3,
    array_to_calibration,
    calibration_to_array,
    cals,
    cpar,
    error_function,
    minimize,
    n_cams,
    np,
    xy_0,
    xy_1,
    xy_2,
    xy_3,
):
    # concatenate all markers
    markers = [np.asarray(xy_0[0], dtype=np.float64), np.asarray(xy_1[0], dtype=np.float64), np.asarray(xy_2[0], dtype=np.float64), np.asarray(xy_3[0], dtype=np.float64)]
    positions = [np.asarray(XYZ_0[0], dtype=np.float64), np.asarray(XYZ_1[0], dtype=np.float64), np.asarray(XYZ_2[0], dtype=np.float64), np.asarray(XYZ_3[0], dtype=np.float64)]

    for c in range(n_cams):
        # print(f" Camera {c}\n")

        XYZ = positions[c]
        xy = markers[c]
        # ID = np.argwhere((XYZ[:,0]>-1))[:,0]

        cal = cals[c]
        # print what you get to see it's still a valid guess
        print(f"Positions of camera {c}: {cal.get_pos()}")
        print(f"Angles: {cal.get_angles()}")

    
        # We could use this step only if we do not have a good
        # initial guess, but we have one from the previous step

    
        # four_points = xy[[0,int(ID.max()/4),int(ID.max()*3/4),ID.max()],:].astype(np.float64) # choose manually
        # ref_pts = XYZ[[0,int(ID.max()/4),int(ID.max()*3/4),ID.max()],:].astype(np.float64)


        # targets = convert_arr_metric_to_pixel(
        #     image_coordinates(ref_pts, cal, cpar.get_multimedia_params()),
        # cpar,
        # )
        # print(f"Before: {four_points - targets}")


        # external_calibration(cal, ref_pts, four_points, cpar)


        x0 = calibration_to_array(cal)
        sol = minimize(error_function, x0, args=(cal, XYZ, xy, cpar), method='Nelder-Mead', tol=1e-13)
        array_to_calibration(sol.x, cal)

        # print(f"Positions: {cal.get_pos()}")
        # print(f"Angles: {cal.get_angles()}")

        cal.write(f'cam{c+1}.ori'.encode(), f'cam{c+1}.addpar'.encode())


        # targets = convert_arr_metric_to_pixel(
        #     image_coordinates(ref_pts, cal, cpar.get_multimedia_params()),
        # cpar,
        # )
        # print(f"After: {four_points - targets}")



        # # we always report all markers reprojection for errors
        # all_markers = [np.loadtxt(_) for _ in allmarkers_files]
        # all_XYZ = all_markers[c][:,2:]
        # all_targets = convert_arr_metric_to_pixel(image_coordinates(all_XYZ, cal, cpar.get_multimedia_params()), cpar)
        # np.savetxt(f'./reprojections/openptv_xy_{case_name}_c{c}.txt', all_targets)

            # px.scatter(x=xy[:,0], y=xy[:,1], color=ID).show()
            # fig = ff.create_quiver(x=xy[:,0], y=xy[:,1], u=targets[:,0]-xy[:,0], v=targets[:,1]-xy[:,1], scale=5)
            # fig.show()

            # Not sure I understand it correctly, we calibrate with 
            # some markers but always compare with the full set


        # Note that we always use allmarkers for comparison:
        # newXYZ, rcm = pixel_to_3d(all_markers, cpar, cals, vpar)


        # XYZ = all_markers[0][:,2:]
        # ID = np.argwhere((XYZ[:,0]>-1))[:,0]

        # newXYZ, rcm = pixel_to_3d(all_markers, cpar, cals, vpar)
        # errors = newXYZ - XYZ

        # print(f" Error rms: {np.sqrt(np.sum(errors**2))}")

        # # print(rcm)

        # newxyz = pd.DataFrame(XYZ, columns=['x','y','z'])
        # newxyz['id'] = ID
        # px.scatter_3d(x=newxyz['x'], y=newxyz['y'], z=newxyz['z'], color=newxyz['id']).show()

    
        # newxyz = pd.DataFrame(newXYZ, columns=['x','y','z'])
        # newxyz['id'] = range(len(newXYZ))
        # px.scatter_3d(x=newxyz['x'], y=newxyz['y'], z=newxyz['z'], color=newxyz['id']).show()

        # np.savetxt(f'openptv_errors_{case_name}.txt', np.hstack([newXYZ, newXYZ- XYZ]))
    return


if __name__ == "__main__":
    app.run()
