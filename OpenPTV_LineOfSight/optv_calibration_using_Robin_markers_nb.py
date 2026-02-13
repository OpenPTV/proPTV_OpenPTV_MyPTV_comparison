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
    # OpenPTV calibration using given markers and liboptv
    """)
    return


@app.cell
def _():
    # liboptv is different from openptv_python cause it's a C library with Cython bindings. requires different installation
    return


@app.cell
def _():
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from scipy.optimize import minimize

    import pathlib, os
    working_path = pathlib.Path.cwd()
    return minimize, np, pd


@app.class_definition
class Parameter:
    cams = [0,1,2,3]
    Vmin = [0,0,0]
    Vmax = [300,300,300]
    N1, N2 = 361, 5


@app.cell
def _(np):
    params = Parameter()

    markers = [np.loadtxt('../proPTV_LineOfSight/markers_c'+str(cam)+'.txt') for cam in params.cams]
    XYZ = markers[0][:,2:]
    xy = markers[0][:,:2]
    ID = np.argwhere((XYZ[:,0]>-1))[:,0]
    return ID, XYZ, xy


@app.cell
def _(ID, XYZ, pd):
    _xyz = pd.DataFrame(XYZ, columns=['x', 'y', 'z'])
    # px.scatter_3d(x=xyz['x'], y=xyz['y'], z=xyz['z'], color=xyz['id']).show()
    _xyz['id'] = ID
    return


@app.cell
def _():
    # First, let's calibrate roughly the cameras
    return


@app.cell
def _(ID, XYZ, pd):
    ref_pts = XYZ[[0, 721, 1409, 1462], :]
    _xyz = pd.DataFrame(ref_pts, columns=['x', 'y', 'z'])
    # px.scatter_3d(x=xyz['x'], y=xyz['y'], z=xyz['z'], color=xyz['id']).show()
    _xyz['id'] = ID[[0, 721, 1409, 1462]]
    return (ref_pts,)


@app.cell
def _():
    cam_id = 1 # or 2,3,4
    return


@app.cell
def _():
    import optv.parameters

    return (optv,)


@app.cell
def _(optv):
    optv.parameters.ControlParams
    return


@app.cell
def _():
    # cal = Calibration().from_file(working_path / "calibration" / f"cam{cam_id}.tif.ori", None)
    # cpar = ControlPar().from_file(working_path / "parameters" / "ptv.par")
    # vpar = VolumePar().from_file(working_path / "parameters" / "criteria.par")

    n_cams = 4

    from optv.parameters import ControlParams, VolumeParams
    cpar = ControlParams(n_cams)
    cpar.read_control_par(b"parameters/ptv.par")

    vpar = VolumeParams()
    vpar.read_volume_par(b"parameters/criteria.par")

    from optv.calibration import Calibration

    # Calibration parameters
    cals = []
    for i_cam in range(n_cams):
        cal = Calibration()
        tmp = cpar.get_cal_img_base_name(i_cam)
        cal.from_file(tmp + b".ori", tmp + b".addpar")
        print(cal.get_pos(), cal.get_angles())
        cals.append(cal)
    return Calibration, cal, cpar


@app.cell
def _(xy):
    four_points = xy[[0,721,1409,1462],:]
    print(f"{four_points = }")
    return (four_points,)


@app.cell
def _(cal, cpar, four_points, ref_pts):
    from optv.orientation import external_calibration, full_calibration

    external_calibration(cal, ref_pts, four_points, cpar)
    cal.get_pos(), cal.get_angles()
    return


@app.cell
def _(cal, cpar, four_points, ref_pts):
    from optv.imgcoord import image_coordinates
    from optv.transforms import convert_arr_metric_to_pixel

    targets = convert_arr_metric_to_pixel(
        image_coordinates(ref_pts, cal, cpar.get_multimedia_params()),
    cpar,
    )
    four_points - targets
    return convert_arr_metric_to_pixel, image_coordinates


@app.cell
def _(Calibration, np):
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

    return array_to_calibration, calibration_to_array


@app.cell
def _(
    array_to_calibration,
    convert_arr_metric_to_pixel,
    image_coordinates,
    np,
):
    def error_function(x, cal, XYZ, xy, cpar):
    
        array_to_calibration(x, cal)

        # print(np.concatenate([
        #     cal.get_pos(),
        #     cal.get_angles(),
        #     cal.get_primary_point(),
        #     cal.get_radial_distortion(),
        #     cal.get_decentering(),
        #     cal.get_affine(),
        # ]))
    
        targets = convert_arr_metric_to_pixel(
            image_coordinates(XYZ, cal, cpar.get_multimedia_params()),
        cpar,
        )
        # err = np.sum(np.abs(xy - targets))
        err = np.sum((xy - targets)**2)
        # print(err)
        return err

    return (error_function,)


@app.cell
def _(cal, calibration_to_array):
    x0 = calibration_to_array(cal)
    print(x0)
    return (x0,)


@app.cell
def _(XYZ, cal, cpar, error_function, minimize, x0, xy):
    sol = minimize(error_function, x0, args=(cal, XYZ, xy, cpar), method='Nelder-Mead', tol=1e-11)
    return (sol,)


@app.cell
def _(sol):
    sol.x
    return


@app.cell
def _(
    array_to_calibration,
    cal,
    convert_arr_metric_to_pixel,
    cpar,
    four_points,
    image_coordinates,
    ref_pts,
    sol,
):
    array_to_calibration(sol.x, cal)
    targets_1 = convert_arr_metric_to_pixel(image_coordinates(ref_pts, cal, cpar.get_multimedia_params()), cpar)
    four_points - targets_1
    return


@app.cell
def _(XYZ, cal, convert_arr_metric_to_pixel, cpar, image_coordinates):
    targets_2 = convert_arr_metric_to_pixel(image_coordinates(XYZ, cal, cpar.get_multimedia_params()), cpar)
    return (targets_2,)


@app.cell
def _(targets_2, xy):
    import plotly.figure_factory as ff
    # px.scatter(x=xy[:,0], y=xy[:,1], color=ID).show()
    fig = ff.create_quiver(x=xy[:, 0], y=xy[:, 1], u=targets_2[:, 0] - xy[:, 0], v=targets_2[:, 1] - xy[:, 1], scale=5)
    fig.show()
    return


@app.cell
def _():
    # cal.write(working_path / "calibration" / "cam{cam_id}_scipy.ori", working_path / "calibration" / "cam{cam_id}_scipy.addpar")
    return


if __name__ == "__main__":
    app.run()
