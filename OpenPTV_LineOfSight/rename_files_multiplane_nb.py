import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    # renaming files
    return


@app.cell
def _():
    import pathlib
    import shutil

    return pathlib, shutil


@app.cell
def _(pathlib):
    pathlib.Path.cwd()
    return


@app.cell
def _(pathlib, shutil):
    for _i in range(1, 5):
        listfiles = list(pathlib.Path('.').glob(f'cam*_plane{_i}*'))
        for _file in listfiles:
            print(_file)
            newfile = f'plane{_i}_' + str(_file).replace(f'_plane{_i}', '')
            print(f'copying {_file}, {newfile}')
            shutil.copyfile(_file, pathlib.Path(newfile))
    return


@app.cell
def _():
    import numpy as np
    for k in range(1, 4):
    # Read the original file
        with open(f'../processed/c0/marker_c0_{k}.txt', 'r') as _file:
            lines = _file.readlines()
        lines = lines[1:]
        new_data = []
        for _i, line in enumerate(lines):  # Remove the header
            columns = np.array(line.split(), dtype=float)
            new_row = f'{_i + 1 + k * 1000:<10}{columns[2]:<10.1f}{columns[4]:<10.1f}{1400 - columns[3]:<10.1f}'
            new_data.append(new_row)  # Prepare the new data
        with open(f'plane_{k}.txt', 'w') as _file:
            _file.write('\n'.join(new_data))  # Write the new data to a new file
    return


@app.cell
def _():
    25*19
    return


if __name__ == "__main__":
    app.run()
