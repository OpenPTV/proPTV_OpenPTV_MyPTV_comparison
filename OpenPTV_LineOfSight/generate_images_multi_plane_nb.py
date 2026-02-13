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
    import numpy as np
    from scipy.optimize import minimize

    import pathlib, os
    working_path = pathlib.Path.cwd()
    return (np,)


@app.cell
def _():
    n_cams = 4

    from optv.parameters import ControlParams, VolumeParams
    cpar = ControlParams(n_cams)
    cpar.read_control_par(b"parameters/ptv.par")
    return VolumeParams, cpar, n_cams


@app.cell
def _(cpar):
    cpar.get_image_size()
    return


@app.cell
def _(VolumeParams):
    vpar = VolumeParams()
    vpar.read_volume_par(b"parameters/criteria.par")
    vpar.get_Zmin_lay(), vpar.get_Zmax_lay()
    return


@app.cell
def _():
    from optv.calibration import Calibration

    return (Calibration,)


@app.cell
def _(Calibration, cpar, n_cams):
    cals = []
    for i_cam in range(n_cams):
        cal = Calibration()
        tmp = cpar.get_cal_img_base_name(i_cam)
        print(tmp)
        cal.from_file(tmp + b".ori", tmp + b".addpar")
        print(cal.get_pos(), cal.get_angles())
        cals.append(cal)
    return (cals,)


@app.cell
def _():
    # from optv.transforms import convert_arr_metric_to_pixel
    # from optv.imgcoord import image_coordinates

    # for plane_z in [0, 350, 700, 1050, 1400]:
    #     for cam in range(n_cams):

    #         # Save the modified data to a new file
    #         data = np.loadtxt(f'calibration/modified_plane_{plane_z}.txt')
    #         targets = convert_arr_metric_to_pixel(
    #             image_coordinates(data[:,1:], cals[cam], cpar.get_multimedia_params()),
    #         cpar,
    #         )

    #         # Combine targets and corresponding XYZ data
    #         combined_data = np.hstack((targets, data[:, 1:]))

    #         # Save to CSV file
    #         # np.savetxt(f'calibration/cam_{cam}_plane_{plane_z}.csv', combined_data, delimiter=',', header='x,y,X,Y,Z', comments='')
    return


@app.cell
def _():
    from flowtracks.io import trajectories_table
    trajects = trajectories_table('/home/user/Dropbox/dataset_25.1.2022/notebooks/trajectories_longerthan50.h5')
    return (trajects,)


@app.cell
def _(np, trajects):
    mins = [np.min(tr.pos(), axis=0) for tr in trajects]
    maxs = [np.max(tr.pos(), axis=0) for tr in trajects]
    mins = np.min(mins, axis=0)
    maxs = np.max(maxs, axis=0)
    (mins, maxs)
    return


@app.cell
def _(np):
    tmp_1 = np.meshgrid(np.arange(-480, 1380, 80), np.arange(-200, 1020, 80))
    tmp_1 = np.array(tmp_1).T.reshape(-1, 2)
    return (tmp_1,)


@app.cell
def _(tmp_1):
    import matplotlib.pyplot as plt
    plt.scatter(tmp_1[:, 0], tmp_1[:, 1])
    return


@app.cell
def _(cals, cpar, np, tmp_1):
    from optv.transforms import convert_arr_metric_to_pixel
    from optv.imgcoord import image_coordinates
    for plane_z in [0, 350, 700, 1050, 1400]:
        for cam in range(4):
            data = np.hstack((tmp_1, np.full((tmp_1.shape[0], 1), plane_z)))  # 4 = n_cams
            targets = convert_arr_metric_to_pixel(image_coordinates(data.astype(np.float64), cals[cam], cpar.get_multimedia_params()), cpar)
            combined_data = np.hstack((targets, data))  # Save the modified data to a new file
            np.savetxt(f'newcal/cam_{cam}_plane_{plane_z}.csv', combined_data, delimiter=',', header='x,y,X,Y,Z', comments='')  # data = np.loadtxt(f'calibration/new_plane_{plane_z}.txt')  # Combine targets and corresponding XYZ data  # Save to CSV file
    return


if __name__ == "__main__":
    app.run()
