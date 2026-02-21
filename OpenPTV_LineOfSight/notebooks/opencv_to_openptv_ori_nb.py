import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import os
    import glob
    import cv2
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    # OpenPTV imports
    try:
        from optv.calibration import Calibration, Exterior, Interior, ap_52
    except ImportError:
        raise ImportError("Please install openptv-python: pip install pyptv")

    return (
        Calibration,
        Exterior,
        Interior,
        ap_52,
        cv2,
        glob,
        least_squares,
        np,
        os,
    )


@app.cell
def _(
    Calibration,
    Exterior,
    Interior,
    ap_52,
    cv2,
    glob,
    least_squares,
    np,
    os,
):

    # ==============================================================================
    # CONFIGURATION
    # ==============================================================================
    NUM_ROWS = 10           # Number of rows in the calibration dot grid
    NUM_COLS = 10           # Number of columns in the calibration dot grid
    DOT_SPACING = 5.0       # Physical distance between dots in mm
    PIXEL_SIZE_MM = 0.005   # Physical size of a pixel on the camera sensor in mm (CRITICAL for OpenPTV)

    # ==============================================================================
    # 1. 3D TARGET DEFINITION (ENFORCING PLANARITY)
    # ==============================================================================
    # By defining a rigid local 3D grid where Z=0 for all points, and ONLY 
    # optimizing the plate's rotation/translation during Bundle Adjustment, 
    # we mathematically guarantee the planarity and exact spacing of the points.
    def create_ideal_target():
        objp = np.zeros((NUM_ROWS * NUM_COLS, 3), np.float32)
        objp[:, :2] = np.mgrid[0:NUM_COLS, 0:NUM_ROWS].T.reshape(-1, 2) * DOT_SPACING
        # Center the target around (0,0,0) for better rotation optimization stability
        objp[:, 0] -= (NUM_COLS - 1) * DOT_SPACING / 2.0
        objp[:, 1] -= (NUM_ROWS - 1) * DOT_SPACING / 2.0
        return objp

    # ==============================================================================
    # 2. IMAGE PROCESSING & DETECTION
    # ==============================================================================
    def extract_image_points(folder_paths):
        """
        Scans 4 folders, detects circle grids, and returns synchronized points.
        """
        print("Detecting calibration dots...")
        all_img_points = [[] for _ in range(len(folder_paths))]

        # Assume images are named sequentially (e.g., img_01.png) in all folders
        image_files = [sorted(glob.glob(os.path.join(f, "*.png")) + glob.glob(os.path.join(f, "*.jpg"))) for f in folder_paths]
        num_frames = min([len(f) for f in image_files])

        # Ensure compatibility with OpenCV 4.x
        blobParams = cv2.SimpleBlobDetector_Params()
        blobParams.filterByArea = True
        blobParams.minArea = 20
        detector = cv2.SimpleBlobDetector_create(blobParams)

        valid_frames = []

        for frame_idx in range(num_frames):
            frame_corners = []
            for cam_idx in range(len(folder_paths)):
                img_path = image_files[cam_idx][frame_idx]
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if frame_idx == 0 and cam_idx == 0:
                    global IMG_SIZE
                    IMG_SIZE = (img.shape[1], img.shape[0]) # (width, height)

                ret, corners = cv2.findCirclesGrid(img, (NUM_COLS, NUM_ROWS), 
                                                   flags=cv2.CALIB_CB_SYMMETRIC_GRID, 
                                                   blobDetector=detector)
                if ret:
                    frame_corners.append(corners)
                else:
                    break # Grid not found in this camera for this frame

            # Only keep frames where the grid is visible in ALL 4 cameras simultaneously
            if len(frame_corners) == len(folder_paths):
                valid_frames.append(frame_idx)
                for cam_idx in range(len(folder_paths)):
                    all_img_points[cam_idx].append(frame_corners[cam_idx])

        print(f"Found valid synchronized grids in {len(valid_frames)} frames.")
        return np.array(all_img_points) # Shape: (Num_Cams, Num_Frames, Num_Points, 1, 2)

    # ==============================================================================
    # 3. GLOBAL BUNDLE ADJUSTMENT
    # ==============================================================================
    def bundle_adjustment_residuals(params, num_cams, num_frames, objp, obs_pts):
        """
        Computes reprojection errors. 
        params structure:
        - First (num_cams * 14) elements: Intrinsics and Extrinsics for each camera
          (fx, fy, cx, cy, k1, k2, p1, p2, rx, ry, rz, tx, ty, tz)
        - Remaining (num_frames * 6) elements: Pose of the calibration plate in each frame
          (rx, ry, rz, tx, ty, tz)
        """
        cam_params = params[:num_cams * 14].reshape((num_cams, 14))
        plate_poses = params[num_cams * 14:].reshape((num_frames, 6))

        residuals = []

        for cam_idx in range(num_cams):
            fx, fy, cx, cy, k1, k2, p1, p2 = cam_params[cam_idx, :8]
            cam_rvec = cam_params[cam_idx, 8:11]
            cam_tvec = cam_params[cam_idx, 11:14]

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            D = np.array([k1, k2, p1, p2, 0.0])

            # Transform World to Camera
            R_cam, _ = cv2.Rodrigues(cam_rvec)

            for frame_idx in range(num_frames):
                plate_rvec = plate_poses[frame_idx, :3]
                plate_tvec = plate_poses[frame_idx, 3:6]

                # Transform Local Plate (Z=0) to Global World Coordinates
                R_plate, _ = cv2.Rodrigues(plate_rvec)
                world_pts = (R_plate @ objp.T).T + plate_tvec

                # Project World points onto Camera Sensor
                proj_pts, _ = cv2.projectPoints(world_pts, cam_rvec, cam_tvec, K, D)
                proj_pts = proj_pts.reshape(-1, 2)

                # Observed points
                obs = obs_pts[cam_idx, frame_idx].reshape(-1, 2)

                residuals.append((obs - proj_pts).ravel())

        return np.concatenate(residuals)

    # ==============================================================================
    # 4. OPENCV TO OPENPTV CONVERSION
    # ==============================================================================
    def convert_and_save_openptv(cam_idx, optimized_cam_params):
        """
        Converts OpenCV optimized parameters (pixels/angles) to OpenPTV 
        photogrammetry formats (mm/Euler) and saves .ori/.addpar files.
        """
        fx, fy, cx, cy, k1, k2, p1, p2, rx, ry, rz, tx, ty, tz = optimized_cam_params

        # 1. Extrinsics Conversion (Camera Pose in World)
        R_cv, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
        t_cv = np.array([tx, ty, tz])

        # Position of camera center in the world: X0 = -R^T * T
        X0, Y0, Z0 = -R_cv.T @ t_cv

        # OpenPTV axes: X right, Y up, Z backwards. OpenCV: X right, Y down, Z forward.
        # Apply a flip to Y and Z axes to map rotation matrices.
        R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R_ptv = R_flip @ R_cv

        # Extract Omega, Phi, Kappa (OpenPTV uses standard photogrammetry Euler angles)
        phi = np.arcsin(-R_ptv[2, 0])
        omega = np.arctan2(R_ptv[2, 1], R_ptv[2, 2])
        kappa = np.arctan2(R_ptv[1, 0], R_ptv[0, 0])

        ext = Exterior(x0=X0, y0=Y0, z0=Z0, omega=omega, phi=phi, kappa=kappa)

        # 2. Intrinsics Conversion (Pixels to Millimeters)
        focal_length_mm = ((fx + fy) / 2.0) * PIXEL_SIZE_MM
        w, h = IMG_SIZE

        # Offset of principal point from the geometric center of the sensor
        xh = (cx - w / 2.0) * PIXEL_SIZE_MM
        yh = -(cy - h / 2.0) * PIXEL_SIZE_MM # Invert Y because OpenPTV metric Y points UP

        intr = Interior(xh=xh, yh=yh, cc=focal_length_mm)

        # 3. Distortion Conversion
        # OpenCV applies distortion in dimensionless space. OpenPTV applies it in mm space.
        # Therefore, scale factors based on pixel size must be applied.
        k1_ptv = k1 / (PIXEL_SIZE_MM ** 2)
        k2_ptv = k2 / (PIXEL_SIZE_MM ** 4)
        p1_ptv = p1 / PIXEL_SIZE_MM
        p2_ptv = p2 / PIXEL_SIZE_MM

        dist = ap_52(k1=k1_ptv, k2=k2_ptv, k3=0.0, p1=p1_ptv, p2=p2_ptv, scx=1.0, she=0.0)

        # 4. Build and Save
        calib = Calibration()
        calib.ext = ext
        calib.int = intr
        calib.added_par = dist

        ori_file = f"cam{cam_idx+1}.ori"
        addpar_file = f"cam{cam_idx+1}.addpar"
        calib.write(ori_file, addpar_file)
        print(f"Saved: {ori_file} and {addpar_file}")

    # ==============================================================================
    # MAIN PIPELINE
    # ==============================================================================
    def main(folder_paths):
        objp = create_ideal_target()
        num_cams = len(folder_paths)

        # 1. Detect points
        obs_pts = extract_image_points(folder_paths)
        num_frames = obs_pts.shape[1]

        if num_frames == 0:
            print("Error: No valid frames found across all cameras.")
            return

        # 2. Initialize Guesses using standard OpenCV CalibrateCamera
        print("Generating initial parameter guesses...")
        init_cam_params = []
        init_plate_poses = np.zeros((num_frames, 6))

        for cam_idx in range(num_cams):
            objpoints = [objp for _ in range(num_frames)]
            imgpoints = [obs_pts[cam_idx, f] for f in range(num_frames)]

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, IMG_SIZE, None, None)

            # Flatten cam params: fx, fy, cx, cy, k1, k2, p1, p2, rx, ry, rz, tx, ty, tz
            params = [mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2], 
                      dist[0,0], dist[0,1], dist[0,2], dist[0,3]]

            # Use first frame to set global camera pose relative to the target
            params.extend(rvecs[0].flatten().tolist())
            params.extend(tvecs[0].flatten().tolist())
            init_cam_params.append(params)

            # We'll initialize the plate poses relative to Camera 1's perspective
            if cam_idx == 0:
                for f in range(num_frames):
                    init_plate_poses[f, :3] = rvecs[f].flatten()
                    init_plate_poses[f, 3:6] = tvecs[f].flatten()

        init_cam_params = np.array(init_cam_params)

        # 3. Setup and Run Global Bundle Adjustment
        print("Running Global Bundle Adjustment (enforcing planarity)...")
        x0 = np.hstack((init_cam_params.flatten(), init_plate_poses.flatten()))

        res = least_squares(
            bundle_adjustment_residuals, x0, method='lm', verbose=2,
            args=(num_cams, num_frames, objp, obs_pts),
            max_nfev=500, ftol=1e-6
        )

        print("\nOptimization Complete.")

        # 4. Extract results and Export to OpenPTV
        optimized_cam_params = res.x[:num_cams * 14].reshape((num_cams, 14))

        print("Exporting to OpenPTV format...")
        for cam_idx in range(num_cams):
            convert_and_save_openptv(cam_idx, optimized_cam_params[cam_idx])

    if __name__ == "__main__":
        # Example usage: provide the paths to the 4 camera folders
        cam_folders = [
            "path/to/cam1_images",
            "path/to/cam2_images",
            "path/to/cam3_images",
            "path/to/cam4_images"
        ]

        # Create dummy folders for demonstration if they don't exist
        for folder in cam_folders:
            os.makedirs(folder, exist_ok=True)

        # main(cam_folders)
        print("Please update 'cam_folders' paths and uncomment 'main(cam_folders)' to run.")
    return


if __name__ == "__main__":
    app.run()
