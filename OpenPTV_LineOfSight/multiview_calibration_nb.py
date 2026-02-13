import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import cv2, sys
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib tk' command supported automatically in marimo

    from scipy import linalg

    return cv2, np, plt


@app.cell
def _():
    from multiview_calibration import DLT, main, Parameter

    return (main,)


@app.cell
def _(main):
    main()
    return


app._unparsable_cell(
    r"""
    # Read back the xy+XYZ files and reconstruct xy_c (per camera, per plane) and P (per plane)
        # Assumes files were written with: np.savetxt(params.markerOutput.format(cam=str(cam)), xyXYZ, header='x,y,X,Y,Z')
        # and that params.cams and params.planes are defined in the notebook.
        params = Parameter()
        xy_c = [None] * len(params.cams)   # will become list of lists: cameras -> planes -> Nx2 arrays
        P = [None] * len(params.planes)    # will become list of arrays: planes -> Nx3 arrays

        # Optional helper: if you have an explicit list of marker counts per plane, set one of these attributes on params:
        # params.plane_marker_counts = [n0, n1, n2, ...]
        counts = getattr(params, 'plane_marker_counts', None) or getattr(params, 'n_markers_per_plane', None)

        for ci, cam in enumerate(params.cams):
            fname = params.markerOutput.format(cam=str(cam))
            try:
                data = np.loadtxt(fname, skiprows=1)   # shape (total_points_across_planes, 5)
            except Exception as e:
                raise IOError(f"Failed to load '{fname}': {e}")

            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != 5:
                raise ValueError(f"File {fname} expected to have 5 columns (x,y,X,Y,Z), found {data.shape[1]}")

            xy_all = data[:, 0:2]
            xyz_all = data[:, 2:5]
            total = data.shape[0]
            n_planes = len(params.planes)

            # Determine per-plane counts
            if counts is not None:
                if sum(counts) != total:
                    raise ValueError(f"Provided plane counts sum {sum(counts)} does not match total rows {total} in {fname}")
                plane_counts = counts
            else:
                # try equal-split fallback
                if total % n_planes == 0:
                    plane_counts = [total // n_planes] * n_planes
                else:
                    raise ValueError(
                        "Cannot infer per-plane marker counts automatically. "
                        "Please set params.plane_marker_counts (list of ints) or ensure each plane has equal number of markers."
                    )

            # split into planes
            splits = np.cumsum([0] + plane_counts)
            cam_xy_planes = []
            for pi in range(n_planes):
                start, end = splits[pi], splits[pi+1]
                cam_xy_planes.append(xy_all[start:end, :])

                # For P we only need to set once (they should be identical across cameras)
                plane_xyz = xyz_all[start:end, :]
                if P[pi] is None:
                    P[pi] = plane_xyz.copy()
                else:
                    # sanity check
                    if not np.allclose(P[pi], plane_xyz, atol=1e-6):
                        raise ValueError(f"Inconsistent 3D coordinates for plane {pi} between cameras (camera {cam})")

            xy_c[ci] = cam_xy_planes

        # xy_c is now list (n_cams) of lists (n_planes) of Nx2 arrays
        # P is list (n_planes) of Nx3 arrays
        print(f"Reconstructed xy_c for {len(params.cams)} cameras and {len(params.planes)} planes.")
        for i, p in enumerate(P):
            print(f" Plane {i}: {p.shape[0]} points")
    """,
    name="_"
)


@app.cell
def _():
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
def _(
    M_0,
    M_1,
    M_2,
    M_3,
    P,
    cv2,
    d_0,
    d_1,
    d_2,
    d_3,
    image_size,
    np,
    plt,
    r_0,
    r_1,
    r_2,
    r_3,
    t_0,
    t_1,
    t_2,
    t_3,
):
    # Generate new_XYZ set of 3D points
    P_array = np.vstack(P)
    # new_XYZ = np.array(np.meshgrid(
    #     np.linspace(P_array[:, 0].min(), P_array[:, 0].max(), 5),
    #     np.linspace(P_array[:, 1].min(), P_array[:, 1].max(), 4),
    #     np.linspace(P_array[:, 2].min(), P_array[:, 2].max(), 3)
    # )).T.reshape(-1, 3)


    x = np.linspace(P_array[:, 0].min(), P_array[:, 0].max(), 3)
    y = np.linspace(P_array[:, 1].min(), P_array[:, 1].max(), 3)
    z = np.linspace(P_array[:, 2].min(), P_array[:, 2].max(), 3)

    X, Y, Z = np.meshgrid(x, y, z)
    new_XYZ = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T



    # Create images for each calibrated camera
    for cam_idx, (r, t, M, d) in enumerate(zip([r_0, r_1, r_2, r_3], [t_0, t_1, t_2, t_3], [M_0, M_1, M_2, M_3], [d_0, d_1, d_2, d_3])):
        imgpoints, _ = cv2.projectPoints(new_XYZ, r[0], t[0], M, d)
        binary_image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
        for point in imgpoints:
            cv2.circle(binary_image, (int(point[0][0]), int(point[0][1])), 7, (255, 255, 255), -1)
            cv2.imwrite(f'cam{cam_idx+1}.tif', binary_image)
    
        plt.figure()
        plt.title(f'Camera {cam_idx}')
        plt.imshow(binary_image, cmap='gray')
        plt.show()
    return (new_XYZ,)


@app.cell
def _(new_XYZ, np):
    import pandas as pd

    # Create a DataFrame with particle numbers and positions
    particle_numbers = np.arange(1, len(new_XYZ) + 1)
    # Reshape new_XYZ to match the order of coordinates
    # new_XYZ_reshaped = new_XYZ.reshape(5, 4, 3, 3).transpose(2, 1, 0, 3).reshape(-1, 3)
    data = np.column_stack((particle_numbers, new_XYZ))
    df_new_XYZ = pd.DataFrame(data, columns=['Particle', 'X', 'Y', 'Z'])

    # Save to a tab-delimited CSV file
    df_new_XYZ.to_csv('new_XYZ.csv', sep='\t', index=False, header=False)
    return (particle_numbers,)


@app.cell
def _(new_XYZ, particle_numbers, plt):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(new_XYZ[:, 0], new_XYZ[:, 1], new_XYZ[:, 2], c='b', marker='o')

    # Annotate each point with its particle number
    for i, txt in enumerate(particle_numbers):
        ax.text(new_XYZ[i, 0], new_XYZ[i, 1], new_XYZ[i, 2], '%d' % txt, size=10, zorder=1, color='k')

    # Set labels
    ax.set_xlabel('X (left to right)')
    ax.set_ylabel('Y (top-down)')
    ax.set_zlabel('Z (into the page)')

    # Adjust the view angle
    ax.view_init(elev=90, azim=-90)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
