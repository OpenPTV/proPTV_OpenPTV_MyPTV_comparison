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
    ## Visualize markers from Robin
    """)
    return


@app.class_definition
class Parameter:
    cams, planes = [0,1,2,3], [1,2,3,4,5,6,12,13,14,15,16,17,18,19,20,21,22,23]
    
    marker_distance, Z0 = (40,40), 0 # X[mm], Y[mm], Z[mm]
    marker_size = (25, 19)  # x, y
    image_size = (2560, 2048)  # x[px], y[px]
    
    Zeros = 2
    markerList = "Ilmenau_xy/c{cam}/marker/c{cam}_{plane}.txt" 
    markerImage = "Ilmenau_xy/c{cam}/c{cam}_{plane}_01.tif"
    markerOutput = "c{cam}_xyXYZ.txt"


@app.cell
def _(np):
    def DLT(P1, P2, P3, P4, xy1, xy2, xy3, xy4):
        """
        Args:
            P1, P2, P3, P4: 3x4 projection matrices for cameras i in (1, 2, 3, 4) so that: xy_i = P_i * XYZ and XYZ=(X,Y,Z,1)
            xy1, xy2, xy3, xy4: 2D image points (x, y) for each camera
    
        Returns:
            3D point (X, Y, Z) in world coordinates
        Meaning:
            Perform DLT with 4 cameras to reconstruct a 3D point with projection matrices of cameras 2,3,4: P2,P3,P4, relative to camera 1.
            The system solves the homogeneous system A * XYZ = 0.
            From a pinhole model we know that xy_i = P_i * XYZ with xy_i = [x',y',w] and x=x'/w and y=y'/w building 2 equations for each camera: 
                    x_i = (P_i*XYZ)_row1 / (P_i*XYZ)_raw3  -> (x_i*P_i_raw1 - P_i_raw3) * XYZ = 0
                    y_i = (P_i*XYZ)_row2 / (P_i*XYZ)_raw3  -> (y_i*P_i_raw2 - P_i_raw3) * XYZ = 0
            This we can build matrix the matrix equation A * XYZ = 0 using all four cameras and the two equations above for each of them.
            Due to noise we do not have A*XYZ=0 instead we have A*XYZ\approx 0 with an non-trivial solution XYZ!=0.
            To find the correct solution we use SVD
            SVD decomposes A = U S V^T, where S is diagonal with singular values.
            The solution we need is the singular vector V_T which corresponding to the smallest singular value (i.e. the last one in numpys output)
            This value minimizes A*XYZ.
        """
        # Construct the 8x4 matrix A using equations from all 4 cameras
        A = np.vstack([
            xy1[1] * P1[2, :] - P1[1, :],  # y1 * P1_3 - P1_2
            P1[0, :] - xy1[0] * P1[2, :],  # P1_1 - x1 * P1_3
            xy2[1] * P2[2, :] - P2[1, :],  # y2 * P2_3 - P2_2
            P2[0, :] - xy2[0] * P2[2, :],  # P2_1 - x2 * P2_3
            xy3[1] * P3[2, :] - P3[1, :],  # y3 * P3_3 - P3_2
            P3[0, :] - xy3[0] * P3[2, :],  # P3_1 - x3 * P3_3
            xy4[1] * P4[2, :] - P4[1, :],  # y4 * P4_3 - P4_2
            P4[0, :] - xy4[0] * P4[2, :]]) # P4_1 - x4 * P4_3
        # Solve using Singular Value Decomposition (SVD)
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        # The corrct 3D point is the last column of Vh 
        XYZ = Vh[-1, 0:3] / Vh[-1, 3]  # Normalize by the homogeneous coordinate
        return XYZ

    return (DLT,)


@app.cell
def _(DLT):
    # from multiview_calibration import DLT, Parameter
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from mpl_toolkits.mplot3d import Axes3D
    import plotly
    params = Parameter()
    X, Y, Z = np.meshgrid(np.arange(0, params.marker_size[0] * params.marker_distance[0], params.marker_distance[0]), -np.arange(0, params.marker_size[1] * params.marker_distance[1], params.marker_distance[1]), np.linspace(params.Z0, params.Z0, 1))
    # copy of main()
    # load parameters 
    XYZ = [np.asarray(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T, dtype=np.float32) for plane in params.planes]
    xy_c, P_c, R_c, ret_c, M_c, d_c, r_c, t_c, pos_c = ([], [], [], [], [], [], [], [], [])
    # define first plane XYZ as reference with Z = 0
    for i, _c in enumerate(params.cams):
        xy = [np.asarray(np.loadtxt(params.markerList.format(cam=_c, plane=str(_t).zfill(params.Zeros)), skiprows=1), dtype=np.float32) for _t in params.planes]
        xy_c.append(xy)
    # initalize a XYZ plane for each plane position equal to the first plane with Z=0
    # we dont know the exact 3D XYZ position of the planes yet
        _ret, _M, _d, _r, _t = cv2.calibrateCamera(XYZ, xy, params.image_size, None, None)
        (ret_c.append(_ret), M_c.append(_M), d_c.append(_d), r_c.append(_r), t_c.append(_t))
    # calibrate cameras individually
        R = cv2.Rodrigues(_r[0])[0]
        R_c.append(R)
        pos = -np.dot(R.T, _t[0]).ravel()
        pos_c.append(pos)
        if i == 0:
            RT = np.concatenate([R, _t[0]], axis=-1)
            P_c.append(_M @ RT)  # estimate rotation matrix
        else:
            _ret, CM0, dist0, CM1, dist1, R, T, E, F = cv2.stereoCalibrate(XYZ[:1], xy_c[0][:1], xy_c[i][:1], M_c[0], d_c[0], M_c[i], d_c[i], params.image_size)
            RT = np.concatenate([R @ R_c[0], R @ t_c[0][0] + T], axis=-1)  # estimate camera position
            P_c.append(_M @ RT)
    P = []
    for _p in tqdm(range(len(params.planes)), desc='DLT'):  # build projection matizes relative to camera 0 
        markers_p = []  # projection matrix for camera 0
        for xy0, xy1, xy2, xy3 in zip(xy_c[0][_p], xy_c[1][_p], xy_c[2][_p], xy_c[3][_p]):
            markers_p.append(DLT(P_c[0], P_c[1], P_c[2], P_c[3], xy0, xy1, xy2, xy3))
        P.append(np.asarray(markers_p, dtype=np.float32))  # projection matrix relative to camera 0 for camera 1, 2 and 3
    P[0] = XYZ[0]  # stereo matching of the FIRST 3D marker plane - use only cam 0 and cam i with i in (1,2,3)
    for i, _c in enumerate(params.cams):  # the assumption of straight lines for light rays inside the medium is needed here because only the first plane 3D positions are known
        _ret, _M, _d, _r, _t = cv2.calibrateCamera(P, xy_c[i], params.image_size, M_c[i], d_c[i], flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        ret_c[i], M_c[i], d_c[i], r_c[i], t_c[i] = (_ret, _M, _d, _r, _t)  # projection matrix for camera i with respect to camera 0
        R = cv2.Rodrigues(_r[0])[0]
        R_c[i] = R
    # compute 3D marker positions using all cameras and the DLT algorithm
        pos = -np.dot(R.T, _t[0]).ravel()
        pos_c[i] = pos
        print('position cam ' + str(_c) + ': ', pos)  # For each plane, reconstruct 3D points using corresponding 2D points from all 4 cameras
    for _c in range(len(params.cams)):
        xyXYZ = np.concatenate([np.append(xy_c[_c][i], P[i], axis=1) for i in range(len(params.planes))])
    # recalibrate cameras with the reconstructed XYZ points of all plates
    # save out Soloff dataset which is the xyXYZ list for each camera containing all planes
        np.savetxt(params.markerOutput.format(cam=str(params.cams[_c])), xyXYZ, header='x,y,X,Y,Z')
    return (
        M_c,
        P,
        R,
        R_c,
        XYZ,
        cv2,
        d_c,
        np,
        params,
        plt,
        pos_c,
        r_c,
        ret_c,
        t_c,
        xy_c,
    )


@app.function
def get_openptv_intrinsics(camera_matrix, image_size, pixel_size_mm):
    """
    Converts OpenCV intrinsic parameters to the OpenPTV format.

    Args:
        camera_matrix (np.ndarray): The 3x3 camera matrix from cv2.calibrateCamera().
        image_size (tuple): The image size (width, height) in pixels.
        pixel_size_mm (float): The physical size of a single pixel in millimeters.

    Returns:
        tuple: A tuple containing (f_mm, xp_mm, yp_mm).
    """
    fx_px = camera_matrix[0, 0]
    fy_px = camera_matrix[1, 1]
    cx_px = camera_matrix[0, 2]  # 1. Extract focal length and principal point in pixels from the camera matrix
    cy_px = camera_matrix[1, 2]
    f_mm = (fx_px + fy_px) / 2.0 * pixel_size_mm
    image_width_px, image_height_px = image_size
    center_x_px = image_width_px / 2.0
    center_y_px = image_height_px / 2.0
    offset_x_px = cx_px - center_x_px  # 2. Calculate the back focal distance in mm
    offset_y_px = cy_px - center_y_px  # We assume square pixels, so we can average fx and fy.
    xp_mm = offset_x_px * pixel_size_mm  # If your pixels are not square, you would need separate pixel_size_x and pixel_size_y.
    yp_mm = offset_y_px * pixel_size_mm
    return (f_mm, xp_mm, yp_mm)  # 3. Calculate the principal point offset in mm from the image center  # Calculate the offset in pixels from the center  # Note: Y points down in image coordinates  # Convert the offset to millimeters  # OpenPTV typically expects a standard coordinate system where Y is up,  # but the principal point offset calculation remains the same.  # The sign convention will depend on the exact definition in your OpenPTV version.  # A positive yp often means the principal point is 'above' the center.


@app.cell
def _(R, cv2, np):
    def convert_to_openptv_format(rvec, tvec):
        """
        Converts a single OpenCV rotation and translation vector into
        the OpenPTV format (x, y, z, omega, phi, kappa).

        Args:
            rvec (np.ndarray): A single rotation vector (3x1) from OpenCV.
            tvec (np.ndarray): A single translation vector (3x1) from OpenCV.

        Returns:
            tuple: A tuple containing (x, y, z, omega, phi, kappa).
                   Angles are in radians, which is standard for OpenPTV.
        """
        rotation_matrix_world_to_cam, _ = cv2.Rodrigues(rvec)
        rotation_matrix_cam_to_world = rotation_matrix_world_to_cam.T
        camera_position = -np.dot(rotation_matrix_cam_to_world, tvec)
        coord_system_transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        final_rotation_matrix = np.dot(coord_system_transform, rotation_matrix_cam_to_world)
        _r = R.from_matrix(final_rotation_matrix)
        omega, phi, kappa = _r.as_euler('xyz', degrees=False)
        _x, _y, _z = camera_position.ravel()
        return (_x, _y, _z, omega, phi, kappa)

    return (convert_to_openptv_format,)


@app.cell
def _(
    M_c,
    P,
    R_c,
    convert_to_openptv_format,
    cv2,
    d_c,
    np,
    params,
    pos_c,
    r_c,
    ret_c,
    t_c,
    xy_c,
):
    PIXEL_SIZE_MM = 0.005
    for i_1, _c in enumerate(params.cams):
        _ret, _M, _d, _r, _t = cv2.calibrateCamera(P, xy_c[i_1], params.image_size, M_c[i_1], d_c[i_1], flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        ret_c[i_1], M_c[i_1], d_c[i_1] = (_ret, _M, _d)
        if _ret:
            rvec_for_cam = _r[0]
            tvec_for_cam = _t[0]
            r_c[i_1] = _r
            t_c[i_1] = _t
            _x, _y, _z, omega, phi, kappa = convert_to_openptv_format(rvec_for_cam, tvec_for_cam)
            pos_c[i_1] = np.array([_x, _y, _z])
            R_c[i_1] = cv2.Rodrigues(rvec_for_cam)[0]
            f_mm, xp_mm, yp_mm = get_openptv_intrinsics(_M, params.image_size, PIXEL_SIZE_MM)
            print(f'------ Camera {_c} OpenPTV Parameters ------')
            print('Extrinsic Parameters:')
            print(f'  Position (x, y, z) [mm]:           ({_x:.4f}, {_y:.4f}, {_z:.4f})')
            print(f'  Orientation (omega, phi, kappa) [rad]: ({omega:.4f}, {phi:.4f}, {kappa:.4f})')
            print('\nIntrinsic Parameters:')
            print(f'  Back Focal Distance (f) [mm]:      {f_mm:.4f}')
            print(f'  Principal Point Offset (xp, yp) [mm]: ({xp_mm:.4f}, {yp_mm:.4f})')
            print('-' * 40 + '\n')
        else:
            print(f'Calibration FAILED for camera {_c}.')
    return


@app.cell
def _(M_c, P, cv2, d_c, np, params, plt, r_c, t_c, xy_c):
    """HERE Chose which camera and which plane the metrics are calculated and plotted for"""
    for _c in range(len(params.cams)):
        for i_2 in range(0, len(params.planes)):
            print('')
            print('2D projection error')
            _p, _ = cv2.projectPoints(P[i_2], r_c[_c][0], t_c[_c][0], M_c[_c], d_c[_c])
            _p = _p.reshape(int(params.marker_size[0] * params.marker_size[1]), 2)
            error_xy = np.linalg.norm(xy_c[_c][i_2] - _p, axis=1)
            print(' Camera ' + str(_c) + ' - Plane ' + str(i_2))
            print('  Mean: ' + str(np.round(np.mean(error_xy), 2)), ' Max: ' + str(np.round(np.max(error_xy), 2)), ' STD: ' + str(np.round(np.std(error_xy), 2)))
            if i_2 == 0:
                plt.figure()
                plt.imshow(cv2.imread(params.markerImage.format(cam=_c, plane=str(params.planes[i_2]).zfill(params.Zeros)), cv2.IMREAD_UNCHANGED), cmap='gray')
                plt.plot(xy_c[_c][i_2][:, 0], xy_c[_c][i_2][:, 1], 'o', c='green', label='marker detection')
                plt.plot(_p[:, 0], _p[:, 1], '+', c='red', label='reprojection')
                plt.legend()
                plt.show()
    return (i_2,)


@app.cell
def _(P, i_2, np, params):
    print('')
    print('3D plane errors')
    horizontal_error_XYZ = np.concatenate([np.diff(P[i_2][n:params.marker_size[1], 0]) for n in range(params.marker_size[0])])
    vertical_error_XYZ = np.concatenate([np.diff(P[i_2][n::params.marker_size[0], 1]) for n in range(params.marker_size[1])])
    print(' Plane ' + str(i_2))
    print('  Horizontal - Mean: ' + str(np.round(np.mean(horizontal_error_XYZ), 2)), ' STD: ' + str(np.round(np.std(horizontal_error_XYZ), 2)))
    # plot 3D positions, take attention on the coordinate system orientation
    print('  Vertical - Mean: ' + str(np.round(np.mean(vertical_error_XYZ), 2)), ' STD: ' + str(np.round(np.std(vertical_error_XYZ), 2)))
    return


@app.cell
def _(P, XYZ, np, params, plt, pos_c):
    _fig = plt.figure(figsize=(12, 12))
    axis = _fig.add_subplot(111, projection='3d')
    (axis.set_xlabel('Z [mm]'), axis.set_ylabel('X [mm]'), axis.set_zlabel('Y [mm]'))
    (axis.set_xlim(-4000, 4000), axis.set_ylim(-4000, 4000), axis.set_zlim(-1500, 880))  # 2.38 x 7.0 m , cameras about 20cm away from the bottom and top plate
    axis.scatter(pos_c[0][2], pos_c[0][0], pos_c[0][1], label='c0', c='blue')
    axis.scatter(pos_c[1][2], pos_c[1][0], pos_c[1][1], label='c1', c='green')
    axis.scatter(pos_c[2][2], pos_c[2][0], pos_c[2][1], label='c2', c='brown')
    axis.scatter(pos_c[3][2], pos_c[3][0], pos_c[3][1], label='c3', c='orange')
    axis.scatter(XYZ[0][:, 2], XYZ[0][:, 0], XYZ[0][:, 1], c='red')
    [axis.scatter(P[i][:, 2], P[i][:, 0], P[i][:, 1], c='black') for i in range(1, len(params.planes))]
    # plot geometry
    _theta = np.linspace(0, 2 * np.pi, 100)
    geometry_down = [3500 * np.cos(_theta), 3500 * np.sin(_theta) + 480, np.zeros_like(_theta) - 1500]
    geometry_up = [3500 * np.cos(_theta), 3500 * np.sin(_theta) + 480, np.zeros_like(_theta) + 880]
    axis.plot(geometry_down[0], geometry_down[1], geometry_down[2], c='black')
    axis.plot(geometry_up[0], geometry_up[1], geometry_up[2], c='black')
    axis.plot(0, 480, -1500, 'x', c='black')
    axis.plot(0, 480, 880, 'x', c='black')
    plt.legend()
    plt.show()
    return geometry_down, geometry_up


@app.cell
def _(P, XYZ, geometry_down, geometry_up, params, pos_c):
    import plotly.graph_objs as go
    _fig = go.Figure()
    _fig.add_trace(go.Scatter3d(x=[pos_c[0][2]], y=[pos_c[0][0]], z=[pos_c[0][1]], mode='markers', marker=dict(size=8, color='blue'), name='c0'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[1][2]], y=[pos_c[1][0]], z=[pos_c[1][1]], mode='markers', marker=dict(size=8, color='green'), name='c1'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[2][2]], y=[pos_c[2][0]], z=[pos_c[2][1]], mode='markers', marker=dict(size=8, color='brown'), name='c2'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[3][2]], y=[pos_c[3][0]], z=[pos_c[3][1]], mode='markers', marker=dict(size=8, color='orange'), name='c3'))
    _fig.add_trace(go.Scatter3d(x=XYZ[0][:, 2], y=XYZ[0][:, 0], z=XYZ[0][:, 1], mode='markers', marker=dict(size=3, color='red'), name='Reference Plane'))
    for i_3 in range(1, len(params.planes)):
        _fig.add_trace(go.Scatter3d(x=P[i_3][:, 2], y=P[i_3][:, 0], z=P[i_3][:, 1], mode='markers', marker=dict(size=2, color='black'), name=f'Plane {i_3}'))
    _fig.add_trace(go.Scatter3d(x=geometry_down[0], y=geometry_down[1], z=geometry_down[2], mode='lines', line=dict(color='black', width=2), name='Geometry Down'))
    _fig.add_trace(go.Scatter3d(x=geometry_up[0], y=geometry_up[1], z=geometry_up[2], mode='lines', line=dict(color='black', width=2), name='Geometry Up'))
    _fig.add_trace(go.Scatter3d(x=[0], y=[480], z=[-1500], mode='markers', marker=dict(symbol='x', size=10, color='black'), name='Bottom Center'))
    _fig.add_trace(go.Scatter3d(x=[0], y=[480], z=[880], mode='markers', marker=dict(symbol='x', size=10, color='black'), name='Top Center'))
    _fig.update_layout(scene=dict(xaxis_title='Z [mm]', yaxis_title='X [mm]', zaxis_title='Y [mm]', xaxis=dict(range=[-4000, 4000]), yaxis=dict(range=[-4000, 4000]), zaxis=dict(range=[-1500, 880])), legend=dict(itemsizing='constant'), width=900, height=900, title='3D Camera and Marker Geometry')
    _fig.show()
    return (go,)


@app.cell
def _(P, XYZ, go, np, params, pos_c):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter3d(x=[pos_c[0][0]], y=[pos_c[0][1]], z=[pos_c[0][2]], mode='markers', marker=dict(size=8, color='blue'), name='c0'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[1][0]], y=[pos_c[1][1]], z=[pos_c[1][2]], mode='markers', marker=dict(size=8, color='green'), name='c1'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[2][0]], y=[pos_c[2][1]], z=[pos_c[2][2]], mode='markers', marker=dict(size=8, color='brown'), name='c2'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[3][0]], y=[pos_c[3][1]], z=[pos_c[3][2]], mode='markers', marker=dict(size=8, color='orange'), name='c3'))
    _fig.add_trace(go.Scatter3d(x=XYZ[0][:, 0], y=XYZ[0][:, 1], z=XYZ[0][:, 2], mode='markers', marker=dict(size=3, color='red'), name='Reference Plane'))
    for i_4 in range(1, len(params.planes)):
        _fig.add_trace(go.Scatter3d(x=P[i_4][:, 0], y=P[i_4][:, 1], z=P[i_4][:, 2], mode='markers', marker=dict(size=2, color='black'), name=f'Plane {i_4}'))
    _theta = np.linspace(0, 2 * np.pi, 100)
    geometry_down_1 = [3500 * np.cos(_theta), np.zeros_like(_theta) - 1500, 3500 * np.sin(_theta) + 480]
    geometry_up_1 = [3500 * np.cos(_theta), np.zeros_like(_theta) + 880, 3500 * np.sin(_theta) + 480]
    _fig.add_trace(go.Scatter3d(x=geometry_down_1[0], y=geometry_down_1[1], z=geometry_down_1[2], mode='lines', line=dict(color='black', width=2), name='Geometry Down'))
    _fig.add_trace(go.Scatter3d(x=geometry_up_1[0], y=geometry_up_1[1], z=geometry_up_1[2], mode='lines', line=dict(color='black', width=2), name='Geometry Up'))
    _fig.add_trace(go.Scatter3d(x=[0], z=[480], y=[-1500], mode='markers', marker=dict(symbol='x', size=10, color='black'), name='Bottom Center'))
    _fig.add_trace(go.Scatter3d(x=[0], z=[480], y=[880], mode='markers', marker=dict(symbol='x', size=10, color='black'), name='Top Center'))
    _fig.update_layout(scene=dict(xaxis_title='x [mm]', yaxis_title='y [mm]', zaxis_title='z [mm]', xaxis=dict(range=[-4000, 4000]), yaxis=dict(range=[-1500, 880]), zaxis=dict(range=[-4000, 4000])), legend=dict(itemsizing='constant'), width=900, height=900, title='3D Camera and Marker Geometry (Data Coordinate System)')
    _fig.show()
    return


@app.cell
def _():
    from pathlib import Path
    params_1 = Parameter()
    file_paths = []
    for ci, _cam in enumerate(params_1.cams):
        fname = params_1.markerOutput.format(cam=str(_cam))
        file_paths.append(fname)
        if not Path(fname).exists():
            raise FileNotFoundError(f'File {fname} does not exist.')
    return Path, file_paths


@app.cell
def _(file_paths):
    file_paths
    return


@app.cell
def _(file_paths):
    import pandas as pd
    from functools import reduce
    with open(file_paths[0], 'r') as _f:
        header = _f.readline().lstrip('#').strip().split(',')
    data_list = []
    for fp in file_paths[:4]:
        with open(fp, 'r') as _f:
            header = _f.readline().lstrip('#').strip().split(',')
        df = pd.read_csv(fp, sep='\\s+', skiprows=1, names=[h.strip() for h in header])
        data_list.append(df)
    for i_5, df in enumerate(data_list):
        df = df.rename(columns={'x': f'x_c{i_5}', 'y': f'y_c{i_5}'})
        data_list[i_5] = df[['X', 'Y', 'Z', f'x_c{i_5}', f'y_c{i_5}']]
    data = reduce(lambda left, right: pd.merge(left, right, on=['X', 'Y', 'Z'], how='inner'), data_list)
    cols = ['X', 'Y', 'Z']
    for i_5 in range(4):
        cols.extend([f'x_c{i_5}', f'y_c{i_5}'])
    data = data[cols]
    data.head()
    return data, df, pd


@app.cell
def _(data, go, np, pos_c):
    X_1 = data['X']
    Y_1 = data['Y']
    Z_1 = data['Z']
    _fig = go.Figure()
    _fig.add_trace(go.Scatter3d(x=[pos_c[0][0]], y=[pos_c[0][1]], z=[pos_c[0][2]], mode='markers', marker=dict(size=8, color='blue'), name='c0'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[1][0]], y=[pos_c[1][1]], z=[pos_c[1][2]], mode='markers', marker=dict(size=8, color='green'), name='c1'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[2][0]], y=[pos_c[2][1]], z=[pos_c[2][2]], mode='markers', marker=dict(size=8, color='brown'), name='c2'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[3][0]], y=[pos_c[3][1]], z=[pos_c[3][2]], mode='markers', marker=dict(size=8, color='orange'), name='c3'))
    _fig.add_trace(go.Scatter3d(x=X_1, y=Y_1, z=Z_1, mode='markers', marker=dict(size=1, color='blue'), name='planes'))
    _theta = np.linspace(0, 2 * np.pi, 100)
    geometry_down_2 = [3500 * np.cos(_theta), np.zeros_like(_theta) - 1500, 3500 * np.sin(_theta) + 480]
    geometry_up_2 = [3500 * np.cos(_theta), np.zeros_like(_theta) + 880, 3500 * np.sin(_theta) + 480]
    _fig.add_trace(go.Scatter3d(x=geometry_down_2[0], y=geometry_down_2[1], z=geometry_down_2[2], mode='lines', line=dict(color='black', width=2), name='Geometry Down'))
    _fig.add_trace(go.Scatter3d(x=geometry_up_2[0], y=geometry_up_2[1], z=geometry_up_2[2], mode='lines', line=dict(color='black', width=2), name='Geometry Up'))
    _fig.add_trace(go.Scatter3d(x=[0], z=[480], y=[-1500], mode='markers', marker=dict(symbol='x', size=10, color='black'), name='Bottom Center'))
    _fig.add_trace(go.Scatter3d(x=[0], z=[480], y=[880], mode='markers', marker=dict(symbol='x', size=10, color='black'), name='Top Center'))
    _fig.update_layout(scene=dict(xaxis_title='x [mm]', yaxis_title='y [mm]', zaxis_title='z [mm]', xaxis=dict(range=[-4000, 4000]), yaxis=dict(range=[-1500, 880]), zaxis=dict(range=[-4000, 4000])), legend=dict(itemsizing='constant'), width=900, height=900, title='3D Camera and Marker Geometry (Data Coordinate System)')
    _fig.show()
    return X_1, Y_1, Z_1


@app.cell
def _(X_1, Y_1, Z_1, data, np):
    def farthest_point_sampling(X, Y, Z, n_samples=10, seed=None):
        rng = np.random.default_rng(seed)
        points = np.stack([X.values, Y.values, Z.values], axis=1)
        n_points = points.shape[0]
        selected_indices = []
        idx = rng.integers(n_points)
        selected_indices.append(idx)
        dists = np.linalg.norm(points - points[idx], axis=1)  # Start with a random point
        for _ in range(1, n_samples):
            idx = np.argmax(dists)
            selected_indices.append(idx)  # Compute distances to the first point
            dists = np.minimum(dists, np.linalg.norm(points - points[idx], axis=1))
        return selected_indices
    fps_indices = farthest_point_sampling(X_1, Y_1, Z_1, n_samples=40, seed=42)
    sampled_fps = data.loc[fps_indices, [f'x_c{i}' for i in range(4)] + [f'y_c{i}' for i in range(4)] + ['X', 'Y', 'Z']]
    # Example usage:
    sampled_fps  # Update distances: for each point, keep the minimum distance to any selected point
    return (fps_indices,)


@app.cell
def _(X_1, Y_1, Z_1, data, fps_indices, go, np, pos_c):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter3d(x=X_1, y=Y_1, z=Z_1, mode='markers', marker=dict(size=2, color='blue'), name='All Markers'))
    sampled_fps_1 = data.loc[fps_indices, ['X', 'Y', 'Z']]
    _fig.add_trace(go.Scatter3d(x=sampled_fps_1['X'], y=sampled_fps_1['Y'], z=sampled_fps_1['Z'], mode='markers', marker=dict(size=8, color='red'), name='Random Sampled Points'))
    _fig.update_layout(scene=dict(xaxis_title='X (left-right)', yaxis_title='Y (upwards)', zaxis_title='Z (depth)'), title='3D Scatter Plot with Random Sampled Points Highlighted')
    _theta = np.linspace(0, 2 * np.pi, 100)
    geometry_down_3 = [3500 * np.cos(_theta), np.zeros_like(_theta) - 1500, 3500 * np.sin(_theta) + 480]
    geometry_up_3 = [3500 * np.cos(_theta), np.zeros_like(_theta) + 880, 3500 * np.sin(_theta) + 480]
    _fig.add_trace(go.Scatter3d(x=geometry_down_3[0], y=geometry_down_3[1], z=geometry_down_3[2], mode='lines', line=dict(color='black', width=2), name='Geometry Down'))
    _fig.add_trace(go.Scatter3d(x=geometry_up_3[0], y=geometry_up_3[1], z=geometry_up_3[2], mode='lines', line=dict(color='black', width=2), name='Geometry Up'))
    _fig.add_trace(go.Scatter3d(x=[0], z=[480], y=[-1500], mode='markers', marker=dict(symbol='x', size=3, color='black'), name='Bottom Center'))
    _fig.add_trace(go.Scatter3d(x=[0], z=[480], y=[880], mode='markers', marker=dict(symbol='x', size=3, color='black'), name='Top Center'))
    _fig.update_layout(scene=dict(xaxis_title='x [mm]', yaxis_title='y [mm]', zaxis_title='z [mm]', xaxis=dict(range=[-4000, 4000]), yaxis=dict(range=[-1500, 880]), zaxis=dict(range=[-4000, 4000])), legend=dict(itemsizing='constant'), width=900, height=900, title='3D Camera and Marker Geometry (Data Coordinate System)')
    _fig.add_trace(go.Scatter3d(x=[pos_c[0][0]], y=[pos_c[0][1]], z=[pos_c[0][2]], mode='markers', marker=dict(size=8, color='blue'), name='c0'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[1][0]], y=[pos_c[1][1]], z=[pos_c[1][2]], mode='markers', marker=dict(size=8, color='green'), name='c1'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[2][0]], y=[pos_c[2][1]], z=[pos_c[2][2]], mode='markers', marker=dict(size=8, color='brown'), name='c2'))
    _fig.add_trace(go.Scatter3d(x=[pos_c[3][0]], y=[pos_c[3][1]], z=[pos_c[3][2]], mode='markers', marker=dict(size=8, color='orange'), name='c3'))
    _fig.show()
    return (sampled_fps_1,)


@app.cell
def _(sampled_fps_1):
    # Prepare data for saving: running index (0-based), X, Y, Z for fps_indices
    calib_block = sampled_fps_1.reset_index(drop=True)
    calib_block.insert(0, 'idx', range(len(calib_block)))
    # Save to tab-delimited text file
    calib_block[['idx', 'X', 'Y', 'Z']].to_csv('cal/new/calibration_block.txt', sep='\t', index=False, header=False, float_format='%.8f')
    return (calib_block,)


@app.cell
def _(data, fps_indices, np, plt):
    from PIL import Image, ImageDraw
    import imageio.v3 as imageio
    camera_names = ['c0', 'c1', 'c2', 'c3']
    for i_6, _cam in enumerate(camera_names):
        img = Image.new('L', (2560, 2048), 0)
        draw = ImageDraw.Draw(img)
        xy_points = data.loc[fps_indices, [f'x_{_cam}', f'y_{_cam}']].values
        xy_points = np.round(xy_points).astype(int)
        for _x, _y in xy_points:
            if 0 <= _x < 2560 and 0 <= _y < 2048:
                draw.ellipse((_x - 5, _y - 5, _x + 5, _y + 5), fill=255)
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray')
        plt.title(f'Sampled 2D Points for Camera {_cam.upper()}')
        plt.axis('off')
        for j, (_x, _y) in enumerate(xy_points):
            plt.text(_x + 10, _y, str(j), color='yellow', fontsize=16, fontweight='bold', va='center', ha='left', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
        plt.show()
        imageio.imwrite(f'./cal/new/cam{i_6 + 1}.tif', np.array(img))
    return


@app.cell
def _(Path, calib_block, go, np):
    pyptv_folder = Path.cwd()
    folder_path = pyptv_folder / 'res'
    cal_path = pyptv_folder / 'cal'
    ori_files = cal_path.rglob('*.ori')
    file_path = f'{folder_path}/rt_is.123456789'
    data_1 = np.loadtxt(file_path, skiprows=1)
    import plotly.express as px
    filtered_data = data_1[np.sum(data_1[:, -4:] == -1, axis=1) < 3]
    _x = filtered_data[:, 1]
    _y = filtered_data[:, 2]
    _z = filtered_data[:, 3]
    _fig = go.Figure(data=[go.Scatter3d(x=_x, y=_y, z=_z, mode='markers')])
    _fig.update_layout(title='3D Scatter Plot', scene=dict(xaxis_title='Z', yaxis_title='X', zaxis_title='Y'))
    for _f in ori_files:
        with open(_f, 'r') as file:
            cam_pos = np.array(file.readline().strip().split(), dtype=float)
            cam_angles = np.array(file.readline().strip().split(), dtype=float)
        direction = np.array([np.cos(cam_angles[1]) * np.cos(cam_angles[0]), np.sin(cam_angles[1]), np.cos(cam_angles[1]) * np.sin(cam_angles[0])])
        _fig.add_trace(go.Scatter3d(x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]], mode='markers', marker=dict(size=5, color='red'), name='Camera Position'))
        _fig.add_trace(go.Scatter3d(x=[cam_pos[0], cam_pos[0] + direction[0]], y=[cam_pos[1], cam_pos[1] + direction[1]], z=[cam_pos[2], cam_pos[2] + direction[2]], mode='lines', line=dict(color='red', width=5), name='Camera Direction'))
    _fig.add_trace(go.Scatter3d(x=calib_block['X'], y=calib_block['Y'], z=calib_block['Z'], mode='markers+text', marker=dict(size=6, color='red', opacity=0.8), text=calib_block['idx'].astype(str), textposition='top center', name='Calibration Block Points'))
    _fig.update_layout(scene=dict(xaxis_title='X (left-right)', yaxis_title='Y (upwards)', zaxis_title='Z (depth)', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0))), title='3D Scatter Plot of Calibration Block Points with Indices')
    _fig.show()
    return (filtered_data,)


@app.cell
def _(calib_block, go):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter3d(x=calib_block['X'], y=calib_block['Y'], z=calib_block['Z'], mode='markers+text', marker=dict(size=6, color='red'), text=calib_block['idx'].astype(str), textposition='top center', name='Calibration Block Points'))
    _fig.update_layout(scene=dict(xaxis_title='X (left-right)', yaxis_title='Y (upwards)', zaxis_title='Z (depth)', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0))), title='3D Scatter Plot of Calibration Block Points with Indices')
    _fig.show()
    return


@app.cell
def _(calib_block):
    calib_block['X'],calib_block['Y'],calib_block['Z']
    return


@app.cell
def _(calib_block, df):
    calib_block, df
    return


@app.cell
def _(filtered_data, pd):
    df_1 = pd.DataFrame.from_records(filtered_data, columns=['ID', 'X', 'Y', 'Z', 'i', 'j', 'k', 'l'])
    return (df_1,)


@app.cell
def _(calib_block, df_1):
    from numpy import isclose
    order = []
    # Find the order of rows in df that matches the order of rows in calib_block based on X, Y, Z
    for _, row in calib_block.iterrows():
        mask = isclose(df_1['X'], row['X'], atol=50) & isclose(df_1['Y'], row['Y'], atol=50) & isclose(df_1['Z'], row['Z'], atol=50)
        idxs = df_1[mask].index
        if len(idxs) > 0:  # Find the index in df where X, Y, Z match (allowing for floating point tolerance)
            order.append(idxs[0])
        else:
            order.append(None)
    order  # or handle as needed  # This list gives the indices in df that match the order of calib_block
    return (order,)


@app.cell
def _(calib_block, df_1, order, pd):
    # Reorder df according to the found order (ignoring None values)
    df_ordered = df_1.loc[order].reset_index(drop=True)
    comparison = pd.DataFrame({'calib_X': calib_block['X'], 'df_X': df_ordered['X'], 'calib_Y': calib_block['Y'], 'df_Y': df_ordered['Y'], 'calib_Z': calib_block['Z'], 'df_Z': df_ordered['Z']})
    # Compare the coordinates in calib_block and df_ordered
    comparison['diff_X'] = comparison['calib_X'] - comparison['df_X']
    comparison['diff_Y'] = comparison['calib_Y'] - comparison['df_Y']
    comparison['diff_Z'] = comparison['calib_Z'] - comparison['df_Z']
    # Calculate differences
    comparison
    return (df_ordered,)


@app.cell
def _(calib_block, df_ordered, np):
    squared_diffs = (calib_block[['X', 'Y', 'Z']].values - df_ordered[['X', 'Y', 'Z']].values) ** 2
    ls_distance = np.sqrt(np.sum(squared_diffs))
    # Compute the sum of squared differences for X, Y, Z columns
    print(f'Least squares distance between the two datasets: {ls_distance:.4f}')
    return


if __name__ == "__main__":
    app.run()
