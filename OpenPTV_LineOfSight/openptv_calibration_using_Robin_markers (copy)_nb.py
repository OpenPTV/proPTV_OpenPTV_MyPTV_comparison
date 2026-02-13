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
    # OpenPTV calibration using given markers and Scipy
    """)
    return


@app.cell
def _():
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from scipy.optimize import minimize

    from openptv_python.calibration import Calibration
    from openptv_python.parameters import ControlPar, VolumePar, OrientPar

    from openptv_python.imgcoord import image_coordinates
    from openptv_python.trafo import arr_metric_to_pixel
    from openptv_python.orientation import external_calibration, full_calibration


    import pathlib, os
    working_path = pathlib.Path.cwd()
    return (
        Calibration,
        ControlPar,
        VolumePar,
        arr_metric_to_pixel,
        external_calibration,
        image_coordinates,
        minimize,
        np,
        pd,
        working_path,
    )


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
    return (cam_id,)


@app.cell
def _(Calibration, ControlPar, VolumePar, cam_id, working_path):
    cal = Calibration().from_file(working_path / "calibration" / f"cam{cam_id}.tif.ori", None)
    cpar = ControlPar().from_file(working_path / "parameters" / "ptv.par")
    vpar = VolumePar().from_file(working_path / "parameters" / "criteria.par")
    return cal, cpar


@app.cell
def _(xy):
    four_points = xy[[0,721,1409,1462],:]
    print(f"{four_points = }")
    return (four_points,)


@app.cell
def _(cal, cpar, external_calibration, four_points, ref_pts):
    external_calibration(cal, ref_pts, four_points, cpar)
    print(f"{cal.ext_par = }")
    return


@app.cell
def _(arr_metric_to_pixel, cal, cpar, four_points, image_coordinates, ref_pts):
    targets = arr_metric_to_pixel(
        image_coordinates(ref_pts, cal, cpar.mm),
    cpar,
    )
    four_points - targets
    return


@app.cell
def _(Calibration, np):
    def array_to_calibration(x:np.ndarray, cal:Calibration) -> None:
        # cal = Calibration()
        cal.set_pos(x[:3])
        cal.set_angles(x[3:6])
        cal.set_primary_point(x[6:9])
        # cal.set_radial_distortion(x[9:12])
        # cal.set_decentering(x[12:14])
        # cal.set_affine_distortion(x[14:])
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
def _(arr_metric_to_pixel, array_to_calibration, image_coordinates, np):
    def error_function(x, cal, XYZ, xy, cpar):
    
        array_to_calibration(x, cal)
    
        targets = arr_metric_to_pixel(
            image_coordinates(XYZ, cal, cpar.mm),
        cpar,
        )
        # err = np.sum(np.abs(xy - targets))
        err = np.sum(xy - targets)
        # print(err)
        return err

    return (error_function,)


@app.cell
def _(XYZ, cal, calibration_to_array, cpar, error_function, minimize, xy):
    x0 = calibration_to_array(cal)
    print(x0)
    sol = minimize(error_function, x0, args=(cal, XYZ, xy, cpar), method='Nelder-Mead', tol=1e-4)
    return (sol,)


@app.cell
def _(
    arr_metric_to_pixel,
    array_to_calibration,
    cal,
    cpar,
    four_points,
    image_coordinates,
    ref_pts,
    sol,
):
    array_to_calibration(sol.x, cal)
    targets_1 = arr_metric_to_pixel(image_coordinates(ref_pts, cal, cpar.mm), cpar)
    four_points - targets_1
    return


@app.cell
def _(XYZ, arr_metric_to_pixel, cal, cpar, image_coordinates):
    targets_2 = arr_metric_to_pixel(image_coordinates(XYZ, cal, cpar.mm), cpar)
    return (targets_2,)


@app.cell
def _(targets_2, xy):
    import plotly.figure_factory as ff
    # px.scatter(x=xy[:,0], y=xy[:,1], color=ID).show()
    fig = ff.create_quiver(x=xy[:, 0], y=xy[:, 1], u=targets_2[:, 0] - xy[:, 0], v=targets_2[:, 1] - xy[:, 1], scale=5)
    fig.show()
    return


@app.cell
def _(cal, working_path):
    cal.write(working_path / "calibration" / "cam{cam_id}_scipy.ori", working_path / "calibration" / "cam{cam_id}_scipy.addpar")
    return


if __name__ == "__main__":
    app.run()
