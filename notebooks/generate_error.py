import matplotlib.pyplot as plt
import numpy as np 
from soloff import *
from scipy.optimize import least_squares


class Parameter:
    cams = [0,1,2,3]
    Vmin = [0,0,0]
    Vmax = [300,300,300]
    N1, N2 = 361, 5

# load parameter
params = Parameter()
# load marker
markers = [np.loadtxt('markers_c'+str(cam)+'.txt') for cam in params.cams]
XYZ = markers[0][:,2:]

ID = np.argwhere((XYZ[:,0]<200) & (XYZ[:,0]>100) & (XYZ[:,1]<200) & (XYZ[:,1]>100))[:,0]
#ID = np.argwhere((XYZ[:,0]>-1))[:,0]

# generate calibration
ax, ay = [], []
for cam in params.cams:
    xyXYZ = markers[cam][ID]
    def dFx(a):
        return F(xyXYZ[:,2:],a) - xyXYZ[:,0]    
    def dFy(a):
        return F(xyXYZ[:,2:],a) - xyXYZ[:,1]
    sx = least_squares(dFx,np.zeros(19),method='trf').x
    sy = least_squares(dFy,np.zeros(19),method='trf').x
    ax.append(sx)
    ay.append(sy)

# get 3D positions from plate
# triangulate 3D position
camPs = [ np.asarray([markers[0][i,:2],markers[1][i,:2],markers[2][i,:2],markers[3][i,:2]]) for i in range(len(markers[0]))]
P = np.asarray([NewtonSoloff_Triangulation(setP, ax, ay, params)[0] for setP in camPs])

# compare error
err = np.linalg.norm(XYZ-P,axis=1)
mean_err = np.mean(err)
std_err = np.std(err)
print('MAE: ', mean_err, ' +- ' , std_err )

axis = plt.figure().add_subplot(111, projection='3d')
axis.scatter(P[:,2], P[:,0], P[:,1],c='red',s=1)
axis.scatter(XYZ[ID,2], XYZ[ID,0], XYZ[ID,1],c='black',s=10)
axis.set_xlabel('Z',fontsize=13), axis.set_ylabel('X',fontsize=13), axis.set_zlabel('Y',fontsize=13)
plt.tight_layout()
plt.show()