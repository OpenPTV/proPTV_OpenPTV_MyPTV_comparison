import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # OpenCV -> OpenPTV Multiplane Calibration Boilerplate

        Goal: start with OpenCV pinhole calibration from multiple dot planes seen by 4 cameras,
        convert the camera poses into OpenPTV orientation (x, y, z, omega, phi, kappa) with
        the origin at a chosen point on plane 1, then write OpenPTV YAML parameters and run
        OpenPTV calibration to refine distortions.

        This notebook is a template. Fill in the TODO sections with your paths and data.
        """
    )
    return


@app.cell
def _():
    # Core dependencies
    import numpy as np
    import pathlib

    # OpenCV for pinhole calibration
    import cv2

    return cv2, np, pathlib


@app.cell
def _(np, pathlib):
    # TODO: update these paths
    data_root = pathlib.Path("/path/to/your/opencv/inputs")
    openptv_root = pathlib.Path("/path/to/your/openptv/project")

    # Expected file layout (example):
    # data_root/
    #   cam0/plane01.txt, cam0/plane02.txt, ...
    #   cam1/plane01.txt, ...
    # Each file contains Nx2 image points (x, y) for the same known 3D points.

    # Plane definitions
    plane_ids = [1, 2, 3, 4, 5]
    num_cams = 4

    # Dot grid spacing in mm
    spacing_x_mm = 40.0
    spacing_y_mm = 40.0

    # Image size (width, height)
    image_size = (2560, 2048)

    return (
        data_root,
        image_size,
        num_cams,
        openptv_root,
        plane_ids,
        spacing_x_mm,
        spacing_y_mm,
    )


@app.cell
def _(np, plane_ids, spacing_x_mm, spacing_y_mm):
    # Build object points for a single plane (Z = 0) in a local plane coordinate system.
    # TODO: set marker grid shape
    grid_nx = 25
    grid_ny = 19

    xs = np.arange(grid_nx, dtype=float) * spacing_x_mm
    ys = np.arange(grid_ny, dtype=float) * spacing_y_mm
    X, Y = np.meshgrid(xs, ys)
    obj_plane = np.vstack([X.ravel(), Y.ravel(), np.zeros(X.size)]).T

    # Each plane will be offset by its Z position in mm
    # TODO: fill plane_zs to match your experiment
    plane_zs = {pid: float((pid - 1) * 50.0) for pid in plane_ids}

    return grid_nx, grid_ny, obj_plane, plane_zs


@app.cell
def _(data_root, np, plane_ids):
    # Load 2D image points for each camera and plane.
    # Each file should contain Nx2 points in the same order as obj_plane.
    # TODO: adjust filename pattern for your data
    def load_image_points(cam_idx: int, plane_id: int) -> np.ndarray:
        fp = data_root / f"cam{cam_idx}" / f"plane{plane_id:02d}.txt"
        return np.loadtxt(fp)

    image_points = []
    for cam_idx in range(4):
        cam_points = []
        for plane_id in plane_ids:
            cam_points.append(load_image_points(cam_idx, plane_id))
        image_points.append(cam_points)

    return image_points, load_image_points


@app.cell
def _(image_points, np, obj_plane, plane_ids, plane_zs):
    # Build 3D object points for each plane.
    object_points = []
    for plane_id in plane_ids:
        pts = obj_plane.copy()
        pts[:, 2] = plane_zs[plane_id]
        object_points.append(pts.astype(np.float32))

    # OpenCV expects a list of planes per camera
    object_points_per_cam = [object_points for _ in range(len(image_points))]

    return object_points, object_points_per_cam


@app.cell
def _(cv2, image_points, image_size, np, object_points_per_cam):
    # Run OpenCV pinhole calibration per camera.
    # Each camera sees multiple planes of dots.
    cam_calib = []
    for cam_idx, cam_image_points in enumerate(image_points):
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points_per_cam[cam_idx],
            cam_image_points,
            image_size,
            None,
            None,
        )
        cam_calib.append((ret, K, dist, rvecs, tvecs))
        print(f"cam{cam_idx}: reproj error {ret}")

    return cam_calib


@app.cell
def _(cv2, np):
    def opencv_pose_to_openptv(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Convert OpenCV rvec/tvec into OpenPTV (x, y, z, omega, phi, kappa).
        Conventions must match OpenPTV coordinate system.
        """
        R_wc, _ = cv2.Rodrigues(rvec)
        R_cw = R_wc.T
        cam_pos = -R_cw @ tvec.reshape(3, 1)

        # TODO: adjust coordinate transform if OpenCV and OpenPTV axes differ.
        # Example: flip Y/Z to match OpenPTV right-handed frame.
        coord = np.eye(3)
        R_openptv = coord @ R_cw

        # Convert rotation matrix to omega, phi, kappa (OpenPTV uses xyz Euler)
        # TODO: validate axis order and signs.
        def rot_to_euler_xyz(Rm: np.ndarray) -> np.ndarray:
            sy = np.sqrt(Rm[0, 0] ** 2 + Rm[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                omega = np.arctan2(Rm[2, 1], Rm[2, 2])
                phi = np.arctan2(-Rm[2, 0], sy)
                kappa = np.arctan2(Rm[1, 0], Rm[0, 0])
            else:
                omega = np.arctan2(-Rm[1, 2], Rm[1, 1])
                phi = np.arctan2(-Rm[2, 0], sy)
                kappa = 0.0
            return np.array([omega, phi, kappa], dtype=float)

        angles = rot_to_euler_xyz(R_openptv)
        pose = np.hstack([cam_pos.ravel(), angles])
        return pose

    return (opencv_pose_to_openptv,)


@app.cell
def _(cam_calib, np, opencv_pose_to_openptv):
    # Convert OpenCV extrinsics to OpenPTV orientation per camera.
    # We use the first plane as world origin.
    # TODO: adjust origin shift to set (0,0,0) to a desired point on plane 1.
    openptv_poses = []
    for cam_idx, (_ret, _K, _dist, rvecs, tvecs) in enumerate(cam_calib):
        # Use plane 1 extrinsics as reference for this camera.
        pose = opencv_pose_to_openptv(rvecs[0], tvecs[0])
        openptv_poses.append(pose)
        print(f"cam{cam_idx}: pose {pose}")

    return openptv_poses


@app.cell
def _(np, openptv_root, openptv_poses):
    # Write OpenPTV orientation files.
    # TODO: replace with your OpenPTV orientation writer of choice.
    # This is a simple placeholder for .ori format (two lines: position, angles).
    for cam_idx, pose in enumerate(openptv_poses):
        pos = pose[:3]
        ang = pose[3:]
        ori_path = openptv_root / "calibration" / f"cam{cam_idx+1}.tif.ori"
        ori_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ori_path, "w") as f:
            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")
            f.write(f"{ang[0]} {ang[1]} {ang[2]}\n")

    return


@app.cell
def _(openptv_root):
    # Write OpenPTV YAML parameters for pyptv.
    # TODO: update the YAML contents to match your experiment.
    yaml_path = openptv_root / "parameters_Run1.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(
            """\
ptv:
  num_cams: 4
  imgsize_x: 2560
  imgsize_y: 2048
  pixel_size_x: 0.0065
  pixel_size_y: 0.0065
  # TODO: set multimedia params, image base names, and cal file basenames
sequence:
  # TODO: set sequence params
criteria:
  # TODO: set volume parameters
"""
        )
    return (yaml_path,)


@app.cell
def _(openptv_root, yaml_path):
    # Run OpenPTV calibration refinement.
    # TODO: replace with your preferred entry point.
    # Example: pyptv uses ParameterManager and calibration routines.
    print(f"YAML ready at: {yaml_path}")
    print("Next step: run OpenPTV calibration with your calibration images and YAML.")
    return


if __name__ == "__main__":
    app.run()
