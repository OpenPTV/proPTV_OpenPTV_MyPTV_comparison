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
    # Plot rt_is.file in 3D
    """)
    return


@app.cell
def _():
    import numpy as np
    from pathlib import Path

    # Provide the folder path where the file is located
    folder_path = '/home/user/Dropbox/3DPTV_Illmenau/2021_03_31_Data_set/res'  # Replace with the actual folder path
    cal_path = Path(folder_path).parent / 'cal'
    ori_files = cal_path.rglob('*.ori')

    # Construct the full file path
    file_path = f'{folder_path}/rt_is.123456789'

    # Read the file using numpy
    data = np.loadtxt(file_path, skiprows=1)


    # print(data)
    return data, np, ori_files


@app.cell
def _(data, np, ori_files):
    import plotly.graph_objs as go
    import plotly.express as px

    # Filter the rows where the last column is not -1
    filtered_data = data[np.sum(data[:, -4:] == -1, axis=1) < 3]

    # Extract the columns for the 3D scatter plot from the filtered data
    x = filtered_data[:, 0]
    y = filtered_data[:, 1]
    z = filtered_data[:, 2]

    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])

    # Set plot title and labels
    fig.update_layout(title='3D Scatter Plot', scene=dict(
        xaxis_title='Z',
        yaxis_title='X',
        zaxis_title='Y'
    ))


    for f in ori_files:
        with open(f, 'r') as file:
            # Read the first line for camera position
            cam_pos = np.array(file.readline().strip().split(), dtype=float)
            # Read the second line for camera angles
            cam_angles = np.array(file.readline().strip().split(), dtype=float)
    
        # Calculate the direction vector from the angles
        direction = np.array([
            np.cos(cam_angles[1]) * np.cos(cam_angles[0]),
            np.sin(cam_angles[1]),
            np.cos(cam_angles[1]) * np.sin(cam_angles[0])
        ])
    
        # Plot the camera position
        fig.add_trace(go.Scatter3d(
            x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Camera Position'
        ))
    
        # Plot the direction arrow
        fig.add_trace(go.Scatter3d(
            x=[cam_pos[0], cam_pos[0] + direction[0]],
            y=[cam_pos[1], cam_pos[1] + direction[1]],
            z=[cam_pos[2], cam_pos[2] + direction[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name='Camera Direction'
        ))

    # Show the plot
    fig.show()
    return


if __name__ == "__main__":
    app.run()
