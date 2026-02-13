import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell
def _():
    # if you need to install openptv-python, it's recommended to use conda

    # Create conda environment with openptv-python
    # conda create -n openptv_lineofsight python=3.10
    # conda activate openptv_lineofsight
    # pip install git+https://github.com/openptv/openptv-python.git
    # conda install jupyter matplotlib numpy numba
    return


@app.cell
def _():
    from openptv_python.calibration import Calibration
    from openptv_python.parameters import ControlPar, VolumePar
    from openptv_python.trafo import pixel_to_metric, dist_to_flat, metric_to_pixel
    from openptv_python.multimed import ray_tracing, move_along_ray
    from openptv_python.imgcoord import img_coord
    import numpy as np
    import matplotlib.pyplot as plt

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
        image_point,
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
def _(subprocess):
    #! ls '/home/user/Downloads/rbc300'
    subprocess.call(['ls', '/home/user/Downloads/rbc300'])
    return


@app.cell
def _():
    # Read the required stuff from the working folder
    import pathlib, os
    working_path = pathlib.Path('/home/user/Downloads/rbc300')
    return os, working_path


@app.cell
def _(os, working_path):
    with open(os.path.join(working_path, 'cal', 'cam1.tif.ori')) as f:
        data = f.read()
    
    print(data)
    return (data,)


@app.cell
def _(data, np):
    data_1 = np.fromstring(data, dtype=float, sep=' ')
    return (data_1,)


@app.cell
def _(data_1):
    data_1
    return


@app.cell
def _(Calibration, ControlPar, VolumePar, os, working_path):
    camera_1_calibration = Calibration().from_file(os.path.join(working_path,"cal","cam1.tif.ori"), None)
    control_parameters = ControlPar().from_file(os.path.join(working_path,"parameters","ptv.par"))
    volume_parameters = VolumePar().from_file(os.path.join(working_path,"parameters","criteria.par"))
    return camera_1_calibration, control_parameters, volume_parameters


@app.cell
def _(camera_1_calibration, control_parameters, volume_parameters):
    camera_1_calibration.ext_par, control_parameters.num_cams, volume_parameters.z_min_lay, volume_parameters.z_max_lay
    return


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
    return (cam_position,)


@app.cell
def _(cam_position):
    cam_position
    return


@app.cell
def _(cam_position, curve_3D, plt):
    # plot line of sight
    axis = plt.figure(figsize=(10,8)).add_subplot(projection='3d')
    axis.set_xlabel('X'), axis.set_ylabel('Y'), axis.set_zlabel('Z')
    axis.set_xlim(0,350), axis.set_ylim(0,350), axis.set_zlim(0,1000)
    axis.set_aspect('equal')

    axis.plot([0,300],[0,0],[0,0],c='black'), axis.plot([0,0],[0,300],[0,0],c='black'), axis.plot([0,0],[0,0],[0,300],c='black')
    axis.plot([300,300],[0,300],[0,0],c='black'), axis.plot([300,300],[0,0],[0,300],c='black'), axis.plot([0,300],[300,300],[0,0],c='black')
    axis.plot([0,0],[300,300],[0,300],c='black'), axis.plot([0,300],[0,0],[300,300],c='black'), axis.plot([0,0],[0,300],[300,300],c='black')
    axis.plot([300,300],[300,300],[0,300],c='black'), axis.plot([300,300],[0,300],[300,300],c='black'), axis.plot([0,300],[300,300],[300,300],c='black')
    # for lof in LOF:
    #     plt.plot( [lof[0,0],lof[0,0]+mu*lof[1,0]] , [lof[0,1],lof[0,1]+mu*lof[1,1]] , [lof[0,2],lof[0,2]+mu*lof[1,2]] ,'-',c='red')
    plt.plot(cam_position[0],cam_position[1],cam_position[2],'o',c='green')
    plt.plot(curve_3D[:,0],curve_3D[:,1],curve_3D[:,2],'-',c='red')
    return


@app.cell
def _(curve_3D):
    curve_3D
    return


@app.cell
def _(Calibration, os, working_path):
    camera_2_calibration = Calibration().from_file(os.path.join(working_path,"cal","cam2.tif.ori"), None)
    camera_2_calibration.ext_par
    return (camera_2_calibration,)


@app.cell
def _(control_parameters):
    control_parameters.mm
    return


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
    fig, ax = plt.subplots(1,2,figsize=(10,8))
    from matplotlib.patches import Rectangle

    ax[0].add_patch(Rectangle((0,0),control_parameters.imx, control_parameters.imy,alpha = 0.5))
    ax[1].add_patch(Rectangle((0,0),control_parameters.imx, control_parameters.imy,alpha = 0.5))
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    for point in curve_3D:    
        x, y = img_coord(point, camera_1_calibration, control_parameters.mm)    
        x, y = metric_to_pixel(x, y, control_parameters)
        # print(x,y)
    
        ax[0].plot(x,y,'ro')
    
    
        x, y = img_coord(point, camera_2_calibration, control_parameters.mm)    
        x, y = metric_to_pixel(x, y, control_parameters)
        # print(x,y)
        ax[1].plot(x,y,'kx')    

    ax[0].set_title('Camera 1')
    ax[1].set_title('Camera 2')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
