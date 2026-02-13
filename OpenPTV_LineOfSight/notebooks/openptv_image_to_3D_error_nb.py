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
    # Error in 3D

    In this notebook we load the points OpenPTV creates from images of calibration plates at different planes and compares them to the ground truth 3D positions
    """)
    return


@app.cell
def _():
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    return go, pd, px


@app.cell
def _(go, pd, px):
    for i in range(1,6):
        df = pd.read_csv(f'./reconstruction/plane{i}.123456789',skiprows=1,sep='\s+',header=None)
        df.columns = ['id','x','y','z','i0','i1','i2','i3']
        # df.head()
        ground = pd.read_csv(f'./calibration/plane_{i}.csv',header=None)
        ground.columns = ['id','x','y','z']
        # ground.head()


        sc = px.scatter_3d(df, x='x', y='y', z='z',color='id')
        fig2 = go.Figure(data=[go.Scatter3d(x=ground.x, y=ground.y, z=ground.z,
                                        mode='markers')])
        fig2.add_traces(sc.data)
        fig2.show()
    return


if __name__ == "__main__":
    app.run()
