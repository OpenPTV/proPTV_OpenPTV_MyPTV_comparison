import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import numpy as np

    # Read the file into a NumPy array
    data = np.loadtxt('plane1/plane_1.txt')

    # Change the distance between points in the 2nd and 3rd columns from 75 mm to 40 mm
    data[:, 1] = (data[:, 1] / 75) * 40
    data[:, 2] = (data[:, 2] / 75) * 40

    # Save the modified data to a new file
    np.savetxt('plane1/modified_plane_1.txt', data, fmt='%3d \t %6.3f \t  %6.3f \t %6.3f')
    return


if __name__ == "__main__":
    app.run()
