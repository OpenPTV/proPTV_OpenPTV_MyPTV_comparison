import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import pandas as pd

    import os
    from glob import glob

    csv_files = glob('/home/user/Downloads/rbc300/cal/plane*.csv')
    csv_files.sort()

    # List of CSV file paths
    # csv_files = ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv']

    # Read CSV files into separate dataframes
    dfs = [pd.read_csv(file, index_col=0,header=None, names=['id','x','y','z']) for file in csv_files]

    # Concatenate dataframes into a single dataframe
    df = pd.concat(dfs, ignore_index=False)

    # Renumber the first column from zero to the number of lines
    # df.reset_index(drop=True, inplace=True)

    # Save the concatenated dataframe as a CSV file
    df.to_csv('/home/user/Downloads/rbc300/cal/all_planes.csv', index=True, header=False)
    return


if __name__ == "__main__":
    app.run()
