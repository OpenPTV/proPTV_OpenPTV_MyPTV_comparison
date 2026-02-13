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
    # copy of generate_error.py from proPTV

    the paper is on overleaf, called Calibration methods for MST.

    Last e-mail from Jan 22, 2025 from Robin, we need to estimate based on interpolated planes but generate errors for all markers
    """)
    return


@app.cell
def _():
    # import matplotlib.pyplot as plt
    import numpy as np 
    from pathlib import Path


    from optv.parameters import ControlParams, VolumeParams
    from optv.calibration import Calibration
    # from optv.orientation import external_calibration, full_calibration

    from optv.imgcoord import image_coordinates
    from optv.transforms import convert_arr_metric_to_pixel, convert_arr_pixel_to_metric, distorted_to_flat
    from optv.orientation import point_positions
    from optv.correspondences import MatchedCoords
    from optv.tracking_framebuf import TargetArray

    # import plotly.express as px
    import plotly.figure_factory as ff
    # import plotly.graph_objects as go
    from scipy.optimize import minimize

    import pandas as pd
    import plotly.express as px

    return (
        Calibration,
        ControlParams,
        MatchedCoords,
        Path,
        TargetArray,
        VolumeParams,
        convert_arr_metric_to_pixel,
        image_coordinates,
        minimize,
        np,
        pd,
        point_positions,
        px,
    )


@app.cell
def _(np):
    # copy of SaveLoad.py
    # import numpy as np

    def LoadMarkerList(cam):
        return np.loadtxt('markers_c'+str(cam)+'.txt')

    def SaveMarkerList(data):
        # Format: x0, y0, x1, y1, x2, y2, x3, y3, X, Y, Z, dX, dY, dZ
        return np.savetxt('markers_error.txt',data)

    return


@app.cell
def _():
    class Parameter:
        cams = [0,1,2,3]
        Vmin = [0,0,0]
        Vmax = [300,300,300]
        N1, N2 = 361, 5

    # load parameter
    params = Parameter()
    return (params,)


@app.cell
def _(Path):
    cases_path = (Path.cwd().parent / 'cases')
    cases_path.exists()
    return (cases_path,)


@app.cell
def _(cases_path):
    cases = list(cases_path.rglob('case_*'))
    order = [1,0,2]
    cases =  [cases[i] for i in order ]
    return (cases,)


@app.cell
def _(cases):
    for _c in cases:
        _list_files = list(_c.rglob('markers*'))
        _list_files.sort()
        print(_list_files)
    return


@app.cell
def _(cases):
    allmarkers_files = list(cases[0].rglob('markers*'))
    allmarkers_files.sort()
    print(allmarkers_files)
    return (allmarkers_files,)


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

    return (
        array_to_calibration,
        calibration_to_array,
        error_function,
        pixel_to_3d,
    )


@app.cell
def _(
    Calibration,
    ControlParams,
    VolumeParams,
    allmarkers_files,
    array_to_calibration,
    calibration_to_array,
    cases,
    convert_arr_metric_to_pixel,
    error_function,
    image_coordinates,
    minimize,
    np,
    params,
    pd,
    pixel_to_3d,
    px,
):
    n_cams = len(params.cams)
    cpar = ControlParams(n_cams)
    cpar.read_control_par(b'parameters/ptv.par')
    vpar = VolumeParams()
    vpar.read_volume_par(b'parameters/criteria.par')
    cals = []
    for i_cam in range(n_cams):
        cal = Calibration()
    # Calibration initial guess 
        tmp = cpar.get_cal_img_base_name(i_cam)
        cal.from_file(tmp + b'.ori', tmp + b'.addpar')
        print(cal.get_pos(), cal.get_angles())
        cals.append(cal)
    for case in cases:
        _list_files = list(case.rglob('markers*'))
        _list_files.sort()
        case_name = case.name.split('case_')[-1]
        print(case_name)
        markers = [np.loadtxt(_) for _ in _list_files]
        if 'allmarkers' in case.name:
            all_markers = markers
        for _c in params.cams:  # if 'allmarkers' not in case.name:
            XYZ = markers[_c][:, 2:]  #     continue
            xy = markers[_c][:, :2]
            ID = np.argwhere(XYZ[:, 0] > -1)[:, 0]
            cal = cals[_c]
            (cal.get_pos(), cal.get_angles())  # 'interpolation', ...
            four_points = xy[[0, int(ID.max() / 4), int(ID.max() * 3 / 4), ID.max()], :]
            ref_pts = XYZ[[0, int(ID.max() / 4), int(ID.max() * 3 / 4), ID.max()], :]  # print([_.name for _ in list_files])
            targets = convert_arr_metric_to_pixel(image_coordinates(ref_pts, cal, cpar.get_multimedia_params()), cpar)
            print(f'Before: {four_points - targets}')  # load marker
            x0 = calibration_to_array(cal)
            sol = minimize(error_function, x0, args=(cal, XYZ, xy, cpar), method='Nelder-Mead', tol=1e-11)
            array_to_calibration(sol.x, cal)
            targets = convert_arr_metric_to_pixel(image_coordinates(ref_pts, cal, cpar.get_multimedia_params()), cpar)
            print(f'After: {four_points - targets}')
            all_markers = [np.loadtxt(_) for _ in allmarkers_files]
            all_XYZ = all_markers[_c][:, 2:]
            all_targets = convert_arr_metric_to_pixel(image_coordinates(all_XYZ, cal, cpar.get_multimedia_params()), cpar)  # print(f" Camera {c}\n")
            np.savetxt(f'./reprojections/openptv_xy_{case_name}_c{_c}.txt', all_targets)
        XYZ = all_markers[0][:, 2:]
        ID = np.argwhere(XYZ[:, 0] > -1)[:, 0]
        newXYZ, rcm = pixel_to_3d(all_markers, cpar, cals, vpar)
        errors = newXYZ - XYZ
        print(f' Error rms: {np.sqrt(np.sum(errors ** 2))}')
        newxyz = pd.DataFrame(XYZ, columns=['x', 'y', 'z'])  # print what you get to see it's still a valid guess
        newxyz['id'] = ID
        px.scatter_3d(x=newxyz['x'], y=newxyz['y'], z=newxyz['z'], color=newxyz['id']).show()
        newxyz = pd.DataFrame(newXYZ, columns=['x', 'y', 'z'])
        newxyz['id'] = range(len(newXYZ))  # We could use this step only if we do not have a good
        px.scatter_3d(x=newxyz['x'], y=newxyz['y'], z=newxyz['z'], color=newxyz['id']).show()  # initial guess, but we have one from the previous step
        np.savetxt(f'openptv_errors_{case_name}.txt', np.hstack([newXYZ, newXYZ - XYZ]))  # choose manually  # external_calibration(cal, ref_pts, four_points, cpar)  # print(x0)  # we always report all markers reprojection for errors  # px.scatter(x=xy[:,0], y=xy[:,1], color=ID).show()  # fig = ff.create_quiver(x=xy[:,0], y=xy[:,1], u=targets[:,0]-xy[:,0], v=targets[:,1]-xy[:,1], scale=5)  # fig.show()  # Not sure I understand it correctly, we calibrate with  # some markers but always compare with the full set  # Note that we always use allmarkers for comparison:  # newXYZ, rcm = pixel_to_3d(all_markers, cpar, cals, vpar)  # print(rcm)
    return XYZ, newXYZ


@app.cell
def _(XYZ, newXYZ, np):
    np.hstack([newXYZ, newXYZ- XYZ])
    return


if __name__ == "__main__":
    app.run()
