import numpy as np
import pandas as pd
import pathlib
from scipy.optimize import minimize

from openptv_python.calibration import Calibration
from openptv_python.parameters import ControlPar, VolumePar

from openptv_python.imgcoord import image_coordinates
from openptv_python.trafo import arr_metric_to_pixel
from openptv_python.orientation import external_calibration



working_path = pathlib.Path.cwd() / "OpenPTV_LineOfSight"
print(f"working from {working_path}")


def array_to_calibration(x:np.ndarray, cal:Calibration) -> None:
    # cal = Calibration()
    cal.set_pos(x[:3])
    cal.set_angles(x[3:6])
    cal.set_primary_point(x[6:9])
    cal.set_radial_distortion(x[9:12])
    cal.set_decentering(x[12:14])
    cal.set_affine_distortion(x[14:])
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
    
    targets = arr_metric_to_pixel(
        image_coordinates(XYZ, cal, cpar.mm),
    cpar,
    )
    err = np.sum(np.abs(xy - targets))
    # print(err)
    return err

class Parameter:
    cams = [0,1,2,3]
    Vmin = [0,0,0]
    Vmax = [300,300,300]
    N1, N2 = 361, 5

params = Parameter()


markers = [np.loadtxt('./proPTV_LineOfSight/markers_c'+str(cam)+'.txt') for cam in params.cams]

for cam_id in range(4):
    XYZ = markers[cam_id][:,2:]
    xy = markers[cam_id][:,:2]
    ID = np.argwhere((XYZ[:,0]>-1))[:,0]

    xyz = pd.DataFrame(XYZ, columns=['x','y','z'])
    xyz['id'] = ID
    # px.scatter_3d(x=xyz['x'], y=xyz['y'], z=xyz['z'], color=xyz['id']).show()

    ref_pts = XYZ[[0,721,1409,1462],:]
    xyz = pd.DataFrame(ref_pts, columns=['x','y','z'])
    xyz['id'] = ID[[0,721,1409,1462]]
    # px.scatter_3d(x=xyz['x'], y=xyz['y'], z=xyz['z'], color=xyz['id']).show()

    cal = Calibration().from_file(working_path / "calibration" / f"cam{cam_id+1}.tif.ori", None)
    cpar = ControlPar().from_file(working_path / "parameters" / "ptv.par")
    vpar = VolumePar().from_file(working_path / "parameters" / "criteria.par")

    four_points = xy[[0,721,1409,1462],:]
    print(f"{four_points = }")

    external_calibration(cal, ref_pts, four_points, cpar)
    print(f"{cal.ext_par = }")

    targets = arr_metric_to_pixel(
        image_coordinates(ref_pts, cal, cpar.mm),
    cpar,
    )
    print(four_points - targets)

    
    x0 = calibration_to_array(cal)
    
    print(error_function(x0, cal, XYZ, xy, cpar))
          
    sol = minimize(error_function, x0, args=(cal, XYZ, xy, cpar), method='Nelder-Mead', tol=1e-4)

    array_to_calibration(sol.x, cal)
    
    print(error_function(sol.x, cal, XYZ, xy, cpar))

    targets = arr_metric_to_pixel(
        image_coordinates(ref_pts, cal, cpar.mm),
    cpar,
    )
    print(four_points - targets)


    targets = arr_metric_to_pixel(
        image_coordinates(XYZ, cal, cpar.mm),
    cpar,
    )


    # import plotly.figure_factory as ff
    # # px.scatter(x=xy[:,0], y=xy[:,1], color=ID).show()
    # fig = ff.create_quiver(x=xy[:,0], y=xy[:,1], u=targets[:,0]-xy[:,0], v=targets[:,1]-xy[:,1], scale=5)
    # fig.show()

    cal.write(working_path / "calibration" / f"cam{cam_id+1}_scipy.ori", working_path / "calibration" / f"cam{cam_id+1}_scipy.addpar")


