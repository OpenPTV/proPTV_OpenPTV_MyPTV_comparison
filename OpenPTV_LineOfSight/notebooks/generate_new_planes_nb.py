import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import numpy as np

    # Read the file into a NumPy array
    data = np.loadtxt('calibration/modified_plane_1.txt')

    for plane_z in [0, 350, 700, 1050, 1400]:
        # Change the distance between points in the 2nd and 3rd columns from 75 mm to 40 mm
        data[:, 3] = plane_z*np.ones(data.shape[0])

        # Save the modified data to a new file
        np.savetxt(f'calibration/new_plane_{plane_z}.txt', data, fmt='%3d \t %6.3f \t  %6.3f \t %6.3f')
    return


if __name__ == "__main__":
    app.run()
