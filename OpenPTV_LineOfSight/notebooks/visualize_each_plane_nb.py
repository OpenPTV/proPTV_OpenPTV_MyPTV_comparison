import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import plotly.express as px

    return pd, px


@app.cell
def _(pd):
    xyz = pd.read_csv('/home/user/Downloads/rbc300/res/rt_is.123456789',delim_whitespace=True,skiprows=1,names=['id','x','y','z','i1','i2','i3','i4'])
    xyz
    return (xyz,)


@app.cell
def _(xyz):
    import numpy as np
    np.abs(xyz['z'] - 274) > 10
    return


@app.cell
def _(px, xyz):
    px.scatter_3d(x=xyz['x'], y=xyz['y'], z=xyz['z'], color=xyz['id']).show()
    return


@app.cell
def _(xyz):
    tmp = xyz['z']-26

    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Histogram(x=tmp, xbins=dict(start=-2, end=2, size=0.1))])
    fig.show()
    return


if __name__ == "__main__":
    app.run()
