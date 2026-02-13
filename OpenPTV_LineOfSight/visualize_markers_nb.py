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


@app.cell
def _():
    # %pip install numpy pandas plotly nbformat scipy scikit-learn matplotlib pillow imageio
    return


@app.cell
def _():
    import numpy as np
    from pathlib import Path
    _folder_path = Path.cwd().parent
    # Provide the folder path where the file is located
    print(f'Folder path: {_folder_path}')  # Replace with the actual folder path
    file_paths = list(_folder_path.rglob('markers*.txt'))
    # cal_path = Path(folder_path).parent / 'cal'
    # ori_files = cal_path.rglob('*.ori')
    # Construct the full file path
    file_paths.sort()
    return Path, file_paths, np


@app.cell
def _(file_paths):
    file_paths
    return


@app.cell
def _(file_paths):
    import pandas as pd
    from functools import reduce
    import plotly.express as p
    with open(file_paths[0], 'r') as _f:
        header = _f.readline().lstrip('#').strip().split(',')
    data_list = []
    for fp in file_paths[:4]:
    # Read all marker files into a list of DataFrames
        with open(fp, 'r') as _f:
            header = _f.readline().lstrip('#').strip().split(',')
        df = pd.read_csv(fp, delim_whitespace=True, skiprows=1, names=[h.strip() for h in header])
        data_list.append(df)
    for _i, df in enumerate(data_list):
        df = df.rename(columns={'x': f'x_c{_i}', 'y': f'y_c{_i}'})
        data_list[_i] = df[['X', 'Y', 'Z', f'x_c{_i}', f'y_c{_i}']]
    # Merge on X, Y, Z (inner join, assuming these are the same across cameras)
    data = reduce(lambda left, right: pd.merge(left, right, on=['X', 'Y', 'Z'], how='inner'), data_list)
    # Add camera-specific x, y columns
    cols = ['X', 'Y', 'Z']
    for _i in range(4):
        cols.extend([f'x_c{_i}', f'y_c{_i}'])
    data = data[cols]
    # Merge all DataFrames on X, Y, Z
    # Reorder columns: X, Y, Z, x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, x_c3, y_c3
    # print(data)
    data.head()
    return data, df, pd


@app.cell
def _(data):
    import plotly.graph_objs as go
    X = data['X']
    # Extract X, Y, Z columns
    Y = data['Y']
    Z = data['Z']
    _fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=2, color='blue'), name='Markers')])
    _fig.update_layout(scene=dict(xaxis_title='X (left-right)', yaxis_title='Y (upwards)', zaxis_title='Z (depth)'), title='3D Scatter Plot of Markers')
    # Create 3D scatter plot
    # Set axis labels: Y upwards, Z depth, X left-right
    _fig.show()
    return X, Y, Z, go


@app.cell
def _(X, Y, Z, data, np):
    def farthest_point_sampling(X, Y, Z, n_samples=10, seed=None):
        rng = np.random.default_rng(seed)
        points = np.stack([X.values, Y.values, Z.values], axis=1)
        n_points = points.shape[0]
        selected_indices = []
        idx = rng.integers(n_points)
        selected_indices.append(idx)
        dists = np.linalg.norm(points - points[idx], axis=1)
        for _ in range(1, n_samples):
            idx = np.argmax(dists)
            selected_indices.append(idx)
            dists = np.minimum(dists, np.linalg.norm(points - points[idx], axis=1))
        return selected_indices
    fps_indices = farthest_point_sampling(X, Y, Z, n_samples=10, seed=42)
    sampled_fps = data.loc[fps_indices, [f'x_c{_i}' for _i in range(4)] + [f'y_c{_i}' for _i in range(4)] + ['X', 'Y', 'Z']]
    sampled_fps
    return (fps_indices,)


@app.cell
def _(X, Y, Z, data, fps_indices, go):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=2, color='blue'), name='All Markers'))
    sampled_fps_1 = data.loc[fps_indices, ['X', 'Y', 'Z']]
    _fig.add_trace(go.Scatter3d(x=sampled_fps_1['X'], y=sampled_fps_1['Y'], z=sampled_fps_1['Z'], mode='markers', marker=dict(size=8, color='red'), name='Random Sampled Points'))
    _fig.update_layout(scene=dict(xaxis_title='X (left-right)', yaxis_title='Y (upwards)', zaxis_title='Z (depth)'), title='3D Scatter Plot with Random Sampled Points Highlighted')
    _fig.show()
    return (sampled_fps_1,)


@app.cell
def _(sampled_fps_1):
    # Prepare data for saving: running index (0-based), X, Y, Z for fps_indices
    calib_block = sampled_fps_1.reset_index(drop=True)
    calib_block.insert(0, 'idx', range(len(calib_block)))
    # Save to tab-delimited text file
    calib_block[['idx', 'X', 'Y', 'Z']].to_csv('calibration_block.txt', sep='\t', index=False, header=False, float_format='%.8f')
    return (calib_block,)


@app.cell
def _(data, fps_indices, np):
    from PIL import Image, ImageDraw
    import imageio.v3 as imageio
    import matplotlib.pyplot as plt
    camera_names = ['c0', 'c1', 'c2', 'c3']
    # Camera names and corresponding columns
    for _i, cam in enumerate(camera_names):
        img = Image.new('L', (2560, 2048), 0)
        draw = ImageDraw.Draw(img)
        xy_points = data.loc[fps_indices, [f'x_{cam}', f'y_{cam}']].values
        xy_points = np.round(xy_points).astype(int)
        for _x, _y in xy_points:  # Get x, y for this camera from sampled_fps indices
            if 0 <= _x < 2560 and 0 <= _y < 2048:
                draw.ellipse((_x - 5, _y - 5, _x + 5, _y + 5), fill=255)
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray')
        plt.title(f'Sampled 2D Points for Camera {cam.upper()}')
        plt.axis('off')  # Show image
        for j, (_x, _y) in enumerate(xy_points):
            plt.text(_x + 10, _y, str(j), color='yellow', fontsize=16, fontweight='bold', va='center', ha='left', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
        plt.show()
        imageio.imwrite(f'{cam}.tif', np.array(img))  # Add running number (idx) next to each point  # Save image
    return


@app.cell
def _(Path, calib_block, go, np):
    pyptv_folder = Path.cwd().parent / 'pyptv'
    _folder_path = pyptv_folder / 'res'
    cal_path = pyptv_folder / 'cal'
    ori_files = cal_path.rglob('*.ori')
    file_path = f'{_folder_path}/rt_is.123456789'
    data_1 = np.loadtxt(file_path, skiprows=1)
    import plotly.express as px
    filtered_data = data_1[np.sum(data_1[:, -4:] == -1, axis=1) < 3]
    _x = filtered_data[:, 1]
    _y = filtered_data[:, 2]
    z = filtered_data[:, 3]
    _fig = go.Figure(data=[go.Scatter3d(x=_x, y=_y, z=z, mode='markers')])
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
