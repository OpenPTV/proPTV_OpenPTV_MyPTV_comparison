import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _(mo):
    mo.md("""
    # OpenCV Camera Calibration Tuning
    This notebook helps you tune parameters for `cv2.findCirclesGrid` using `cv2.SimpleBlobDetector`.

    1. **Load Image**: Ensure your image path is correct.
    2. **Tune Blob Detector**: Adjust thresholds and filters until blobs (circles) are consistently detected (red circles).
    3. **Find Grid**: Set the grid pattern size (e.g. 21x17) and check if the grid is found (colored corners).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import cv2
    import sys
    import subprocess
    import matplotlib.pyplot as plt
    from skimage import io
    import os
    import numpy as np

    return cv2, mo, np, os, plt


@app.cell
def _(cv2, mo, np, os):
    img_path = "/home/user/Dropbox/3DPTV_Illmenau/00000093_0000000018B72D72.png"
    print(f"Path exists: {os.path.exists(img_path)}")

    if os.path.exists(img_path):
        img_cv = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img_cv = np.zeros((100, 100), dtype=np.uint8)

    img_disp = img_cv.copy()

    mo.md(f"Loaded image: `{img_path}` shape={img_cv.shape}")
    return (img_cv,)


@app.cell
def _(mo):
    # Configuration UI
    mo.md("## Configuration")

    # Blob Detector
    min_thresh = mo.ui.slider(0, 255, value=10, label="Min Threshold")
    max_thresh = mo.ui.slider(0, 255, value=200, label="Max Threshold")
    filter_area = mo.ui.checkbox(value=True, label="Filter by Area")
    min_area = mo.ui.number(start=0, stop=10000, value=10, label="Min Area")
    max_area = mo.ui.number(start=0, stop=100000, value=5000, label="Max Area")
    filter_circ = mo.ui.checkbox(value=False, label="Filter by Circularity")
    min_circ = mo.ui.slider(
        0.0, 1.0, value=0.1, step=0.01, label="Min Circularity"
    )
    filter_conv = mo.ui.checkbox(value=True, label="Filter by Convexity")
    min_conv = mo.ui.slider(0.0, 1.0, value=0.87, step=0.01, label="Min Convexity")
    filter_inert = mo.ui.checkbox(value=True, label="Filter by Inertia")
    min_inert = mo.ui.slider(0.0, 1.0, value=0.01, step=0.01, label="Min Inertia")

    # Grid Pattern
    grid_rows = mo.ui.number(start=3, stop=50, value=21, label="Rows")
    grid_cols = mo.ui.number(start=3, stop=50, value=17, label="Cols")
    flags_clustering = mo.ui.checkbox(value=False, label="Use Clustering")

    ui = mo.vstack(
        [
            mo.md("### Blob Detector Settings"),
            mo.hstack([min_thresh, max_thresh]),
            mo.hstack([filter_area, min_area, max_area]),
            mo.hstack([filter_circ, min_circ]),
            mo.hstack([filter_conv, min_conv]),
            mo.hstack([filter_inert, min_inert]),
            mo.md("### Grid Pattern Settings"),
            mo.hstack([grid_rows, grid_cols, flags_clustering]),
        ]
    )
    # ui
    return (
        filter_area,
        filter_circ,
        filter_conv,
        filter_inert,
        flags_clustering,
        grid_cols,
        grid_rows,
        max_area,
        max_thresh,
        min_area,
        min_circ,
        min_conv,
        min_inert,
        min_thresh,
        ui,
    )


@app.cell
def _(
    cv2,
    filter_area,
    filter_circ,
    filter_conv,
    filter_inert,
    img_cv,
    max_area,
    max_thresh,
    min_area,
    min_circ,
    min_conv,
    min_inert,
    min_thresh,
):
    # Processing
    # Setup Blob Detector
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = min_thresh.value
    params.maxThreshold = max_thresh.value
    params.filterByArea = filter_area.value
    params.minArea = min_area.value
    params.maxArea = max_area.value
    params.filterByCircularity = filter_circ.value
    params.minCircularity = min_circ.value
    params.filterByConvexity = filter_conv.value
    params.minConvexity = min_conv.value
    params.filterByInertia = filter_inert.value
    params.minInertiaRatio = min_inert.value

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_cv)
    print(keypoints)
    return detector, keypoints


@app.cell
def _(cv2, img_cv, keypoints, np):
    # Draw detected blobs
    im_with_keypoints = cv2.drawKeypoints(
        img_cv,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return (im_with_keypoints,)


@app.cell
def _(
    cv2,
    detector,
    flags_clustering,
    grid_cols,
    grid_rows,
    im_with_keypoints,
    img_cv,
    keypoints,
):
    # Find Circles Grid
    pattern_size = (grid_cols.value, grid_rows.value)
    flags = cv2.CALIB_CB_SYMMETRIC_GRID
    if flags_clustering.value:
        flags |= cv2.CALIB_CB_CLUSTERING

    found, corners = cv2.findCirclesGrid(
        img_cv, pattern_size, flags=flags, blobDetector=detector
    )

    # Draw Grid
    if found:
        cv2.drawChessboardCorners(im_with_keypoints, pattern_size, corners, found)
        status_msg = f"**Success!** Grid found ({len(corners)} points)."
    else:
        status_msg = f"**Grid Not Found.** Detected {len(keypoints)} blobs. Adjust parameters."
    return (found,)


@app.cell
def _(found, im_with_keypoints, keypoints, mo, plt, ui):
    plt.figure(figsize=(10, 10))
    plt.imshow(im_with_keypoints)
    plt.title(
        f"Blobs: {len(keypoints)} | Grid: {'Found' if found else 'Not Found'}"
    )
    plt.axis("off")
    plot = plt.gca()

    mo.vstack([ui, plot])
    return


if __name__ == "__main__":
    app.run()
