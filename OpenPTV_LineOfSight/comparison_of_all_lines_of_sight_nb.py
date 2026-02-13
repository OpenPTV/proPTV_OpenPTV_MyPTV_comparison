import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    from openptv_python.calibration import Calibration
    from openptv_python.parameters import ControlPar, VolumePar
    from openptv_python.trafo import pixel_to_metric, dist_to_flat, metric_to_pixel
    from openptv_python.multimed import ray_tracing, move_along_ray
    from openptv_python.imgcoord import img_coord
    import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib widget
    return (
        Calibration,
        ControlPar,
        VolumePar,
        dist_to_flat,
        img_coord,
        metric_to_pixel,
        move_along_ray,
        np,
        pixel_to_metric,
        plt,
        ray_tracing,
    )


@app.cell
def _(
    Calibration,
    ControlPar,
    VolumePar,
    dist_to_flat,
    move_along_ray,
    np,
    pixel_to_metric,
    ray_tracing,
):
    # openptv way to get the line in 3D from the point in the image space and 
    # the calibration parameters


    def epipolar_curve_in_3D(
        image_point: np.ndarray,
        origin_cam: Calibration,
        num_points: int,
        cparam: ControlPar,
        vparam: VolumePar,
    ) -> np.ndarray:
        """
        Get the points lying on the epipolar line from one camera to the other, on.

        the edges of the observed volume. Gives pixel coordinates.

        Assumes the same volume applies to all cameras.

        Arguments:
        ---------
        image_point - the 2D point on the image
            plane of the camera seeing the point. Distorted pixel coordinates.
        Calibration origin_cam - current position and other parameters of the
            camera seeing the point.
        int num_points - the number of points to generate along the line. Minimum
            is 2 for both endpoints.
        ControlParams cparam - an object holding general control parameters.
        VolumeParams vparam - an object holding observed volume size parameters.

        Returns
        -------
        line_points - (num_points,2) array with projection camera image coordinates
            of points lying on the ray stretching from the minimal Z coordinate of
            the observed volume to the maximal Z thereof, and connecting the camera
            with the image point on the origin camera.
        """

        line_points = np.empty((num_points, 3))

        # Move from distorted pixel coordinates to straight metric coordinates.
        x, y = pixel_to_metric(image_point[0], image_point[1], cparam)
        x, y = dist_to_flat(x, y, origin_cam, 0.00001)

        vertex, direct = ray_tracing(x, y, origin_cam, cparam.mm)

        for pt_ix, Z in enumerate(
            np.linspace(vparam.z_min_lay[0], vparam.z_max_lay[0], num_points)
        ):
            # x = line_points[pt_ix], 0)
            # y = <double *>np.PyArray_GETPTR2(line_points, pt_ix, 1)

            line_points[pt_ix, :] = move_along_ray(Z, vertex, direct)
        
            # x, y = img_coord(pos, project_cam, cparam.mm)
            # line_points[pt_ix, 0], line_points[pt_ix, 1] = metric_to_pixel(x, y, cparam)

        return line_points

    return (epipolar_curve_in_3D,)


@app.cell
def _():
    # Read the required stuff from the working folder
    import pathlib, os
    working_path = pathlib.Path('OpenPTV_LineOfSight')
    return os, working_path


@app.cell
def _(Calibration, ControlPar, VolumePar, os, working_path):
    camera_1_calibration = Calibration().from_file(os.path.join(working_path,"calibration","cam1.tif.ori"), None)
    control_parameters = ControlPar().from_file(os.path.join(working_path,"parameters","ptv.par"))
    volume_parameters = VolumePar().from_file(os.path.join(working_path,"parameters","criteria.par"))
    return camera_1_calibration, control_parameters, volume_parameters


@app.cell
def _(
    camera_1_calibration,
    control_parameters,
    epipolar_curve_in_3D,
    volume_parameters,
):
    curve_3D = epipolar_curve_in_3D(
        image_point=[control_parameters.imx/2, control_parameters.imy/2],
        origin_cam = camera_1_calibration,
        num_points = 10,
        cparam = control_parameters,
        vparam = volume_parameters,
    )
    return (curve_3D,)


@app.cell
def _(camera_1_calibration):
    cam_position = camera_1_calibration.get_pos()
    return


@app.cell
def _(Calibration, os, working_path):
    camera_2_calibration = Calibration().from_file(os.path.join(working_path,"calibration","cam2.tif.ori"), None)
    return (camera_2_calibration,)


@app.cell
def _(
    camera_1_calibration,
    camera_2_calibration,
    control_parameters,
    curve_3D,
    img_coord,
    metric_to_pixel,
    plt,
):
    fig, _ax = plt.subplots(1, 2, figsize=(10, 8))
    from matplotlib.patches import Rectangle
    _ax[0].add_patch(Rectangle((0, 0), control_parameters.imx, control_parameters.imy, alpha=0.5))
    _ax[1].add_patch(Rectangle((0, 0), control_parameters.imx, control_parameters.imy, alpha=0.5))
    _ax[0].set_aspect('equal')
    _ax[1].set_aspect('equal')
    for point in curve_3D:
        x, y = img_coord(point, camera_1_calibration, control_parameters.mm)
        x, y = metric_to_pixel(x, y, control_parameters)
        _ax[0].plot(x, y, 'ro')
        x, y = img_coord(point, camera_2_calibration, control_parameters.mm)
        x, y = metric_to_pixel(x, y, control_parameters)  # print(x,y)
        _ax[1].plot(x, y, 'kx')
    _ax[0].set_title('Camera 1')
    _ax[1].set_title('Camera 2')
    plt.show()  # print(x,y)
    return


@app.cell
def _():
    # Now we try to create the line of sight using myptv 
    from myptv.imaging_mod import camera_wrapper, img_system

    # ThreeDModel = 'ExtendedZolof'     # <-- set either to 'Tsai' or to 'ExtendedZolof'
    # ThreeDModel = 'Tsai'     # <-- set either to 'Tsai' or to 'ExtendedZolof'
    cam_name = 'cam0'        # <-- set to one of: cam0, cam1, cam2, cam3
    # directory = '../MyPTV_LineOfSight/cam_Tsai'
    directory = 'MyPTV_LineOfSight/cam_extendedZolof'
    cam = camera_wrapper(cam_name, directory)
    cam.load()

    print(cam)
    return (cam,)


@app.cell
def _(cam, np, plt):
    class LineOfSight(object):
    
        def __init__(self, cam):
        
            self.cam = cam
        
    
        def get_line(self, x, y):
            P, v = self.cam.get_epipolarline(x, y)
            return P, v
    
    
        def plot_line(self, x, y):
            P, v = self.get_line(x, y)
        
        
            axis = plt.figure(figsize=(10,8)).add_subplot(projection='3d')
            axis.set_xlabel('X'), axis.set_ylabel('Y'), axis.set_zlabel('Z')
            axis.set_xlim(-50,350), axis.set_ylim(-50,350), axis.set_zlim(0,1200)

            axis.set_box_aspect([axis.get_xlim()[1] - axis.get_xlim()[0],
                                 axis.get_ylim()[1] - axis.get_ylim()[0],
                                 axis.get_zlim()[1] - axis.get_zlim()[0]])

            axis.plot([0,300],[0,0],[0,0],c='black'), axis.plot([0,0],[0,300],[0,0],c='black'), axis.plot([0,0],[0,0],[0,300],c='black')
            axis.plot([300,300],[0,300],[0,0],c='black'), axis.plot([300,300],[0,0],[0,300],c='black'), axis.plot([0,300],[300,300],[0,0],c='black')
            axis.plot([0,0],[300,300],[0,300],c='black'), axis.plot([0,300],[0,0],[300,300],c='black'), axis.plot([0,0],[0,300],[300,300],c='black')
            axis.plot([300,300],[300,300],[0,300],c='black'), axis.plot([300,300],[0,300],[300,300],c='black'), axis.plot([0,300],[300,300],[300,300],c='black')


            plt.plot(P[0],P[1],P[2],'o',c='green')
        
            curve_3D = np.array([P+10*v])
            for a_ in np.linspace(0, -P[2]/v[2], num=10):
                curve_3D = np.concatenate((curve_3D, np.array([P + a_*v])))
        
            plt.plot(curve_3D[:,0],curve_3D[:,1],curve_3D[:,2],'-',c='red')
        
        
        
        

    LOS = LineOfSight(cam)
    return (LOS,)


@app.cell
def _(LOS, control_parameters):
    LOS.get_line(control_parameters.imx/2, control_parameters.imy/2)
    return


@app.cell
def _(
    camera_1_calibration,
    control_parameters,
    epipolar_curve_in_3D,
    volume_parameters,
):
    cam_position_1 = camera_1_calibration.get_pos()
    curve_3D_1 = epipolar_curve_in_3D(image_point=[control_parameters.imx / 2, control_parameters.imy / 2], origin_cam=camera_1_calibration, num_points=10, cparam=control_parameters, vparam=volume_parameters)
    return cam_position_1, curve_3D_1


@app.cell
def _(cam_position_1, control_parameters, np):
    import sys
    sys.path.append('proPTV_LineOfSight')
    from func import proPTV_LineOfSight, Get_Closest_Point
    calibration_path = 'proPTV_LineOfSight/calibration/c{cam}/soloff_c{cam}{xy}.txt'
    Vmin, Vmax = ([0, 0, 0], [300, 300, 300])
    c = 0
    xy = np.array([[control_parameters.imx / 2, control_parameters.imy / 2], [0, 0], [2100, 0]])
    mu = 5
    _ax, ay = (np.loadtxt(calibration_path.format(cam=c, xy='x'), delimiter=','), np.loadtxt(calibration_path.format(cam=c, xy='y'), delimiter=','))
    LOF = [proPTV_LineOfSight(p, c, Vmin, Vmax, _ax, ay) for p in xy]
    cam_position_proPTV = Get_Closest_Point(LOF)
    print('estimated cam position: ', cam_position_1)
    return LOF, cam_position_proPTV, mu


@app.cell
def _(cam, cam_position_proPTV, camera_1_calibration):
    print(cam_position_proPTV)
    print(f'proPTV: {[cam_position_proPTV[0],cam_position_proPTV[2],-cam_position_proPTV[1]]}')
    print(f'openPTV: {camera_1_calibration.get_pos()}')
    print(f'myPTV: {cam}')
    return


@app.cell
def _(
    LOF,
    LOS,
    cam_position_1,
    cam_position_proPTV,
    control_parameters,
    curve_3D_1,
    mu,
    plt,
):
    # 1. MyPTV:
    LOS.plot_line(control_parameters.imx / 2, control_parameters.imy / 2)
    plt.plot(cam_position_1[0], cam_position_1[1], cam_position_1[2], 'o', c='magenta')
    # 2. openptv-python
    plt.plot(curve_3D_1[:, 0], curve_3D_1[:, 1], curve_3D_1[:, 2], '-', c='blue')
    plt.plot(cam_position_proPTV[0], cam_position_proPTV[2], -cam_position_proPTV[1], 'x', c='black')
    for lof in LOF[:1]:
    # 3. proPTV
    # mu is not clear to me. the angles I think i fixed. 
    # for lof in LOF[:1]: # only first point
    #     plt.plot( [lof[0,0],lof[0,0]+mu*lof[1,0]] , 1*np.array([lof[0,2],lof[0,2]+mu*lof[1,2]]), -1*np.array([lof[0,1],lof[0,1]+mu*lof[1,1]]) ,'-',c='magenta')
    # copied from LineOfSight.main()
        plt.plot([lof[0, 0], lof[0, 0] + mu * lof[1, 0]], [lof[0, 1], lof[0, 1] + mu * lof[1, 1]], [lof[0, 2], lof[0, 2] + mu * lof[1, 2]], '--', c='red')
    return


if __name__ == "__main__":
    app.run()
