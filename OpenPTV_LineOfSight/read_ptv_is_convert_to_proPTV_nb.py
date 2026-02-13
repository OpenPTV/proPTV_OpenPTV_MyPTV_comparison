import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from flowtracks.io import trajectories_ptvis

    from tqdm import tqdm
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.colors import Normalize

    return Line3DCollection, Normalize, np, plt, tqdm, trajectories_ptvis


@app.cell
def _(trajectories_ptvis):
    inName = '~/Downloads/1024_15/res/ptv_is.%d'
    trajects = trajectories_ptvis(inName, traj_min_len=8, xuap=False)
    return (trajects,)


@app.cell
def _(plt, trajects):
    for _traj in trajects:
        plt.plot(_traj.pos()[:, 0], _traj.pos()[:, 1], '.:')
    return


@app.cell
def _(plt, trajects):
    for _traj in trajects:
        plt.plot(_traj.pos()[:, 0], _traj.pos()[:, 2], '.:')
    return


@app.cell
def _(trajects):
    allTracks = []
    for tr in trajects:
        t = tr.time()[:-2]
        _pos = tr.pos()[:-2]
        _vel = tr.velocity()[:-2]
        acc = tr.accel()[:-2]
    # allTracks = list(zip(t,pos,vel,acc))
        allTracks.append([t.tolist(), [row for row in _pos], [row for row in _vel], [row for row in acc]])  # maxvel = max(np.linalg.norm(vel))
    return (allTracks,)


@app.cell
def _(Line3DCollection, Normalize, allTracks, np, plt, tqdm):
    # def PlotTracks(allTracks,maxvel):
    # plot tracks
    maxvel = 0.0001
    fig = plt.figure(figsize=(8, 8))
    axis = fig.add_subplot(111, projection='3d')
    for track in tqdm(allTracks, desc='Plot tracks', position=0, leave=True, delay=0.5):
        _pos = np.asarray(track[1])  # for each track get the positions
        x, y, z = (_pos[:, 0], _pos[:, 1], _pos[:, 2])
        _vel = np.asarray(track[2])
        color = _vel[:, 2]  # load the velocitiy of the track and select the color as the vz velocity
        colorNorm = Normalize(-maxvel, maxvel)
        points = np.array([x, y, z]).transpose().reshape(-1, 1, 3) * 1000
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segs, cmap='seismic', norm=colorNorm, linewidths=0.4, alpha=1)  # define line segment
        lc.set_array(color)
        axis.add_collection3d(lc)
    axis.set_title('number of tracks: ' + str(len(allTracks)))
    (axis.set_xlabel('X'), axis.set_ylabel('Y'), axis.set_zlabel('Z'))
    (plt.tight_layout(), plt.show())  # add the line segment to the figure  # save tracks  # plt.savefig('alexTracks.png',dpi=200)
    return


if __name__ == "__main__":
    app.run()
