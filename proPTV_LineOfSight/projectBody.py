import numpy as np
import matplotlib.pyplot as plt

from func import *
from generateBody import *
import os

def main(parent_path):
    # user defined parameter
    calibration_path = os.path.join(parent_path, "calibration/c{cam}/soloff_c{cam}{xy}.txt")
    c = 0 # camera ID
    
    # load calibration of camera c
    ax, ay = np.loadtxt(calibration_path.format(cam=c,xy="x"),delimiter=','), np.loadtxt(calibration_path.format(cam=c,xy="y"),delimiter=',')
    
    # estimate the 3d body 
    torus, mid, spiral = Torus(100, 40, 100, 8, 150, np.pi/2, np.array([0,0,1]))
    X,Y,Z = torus
    X_mid,Y_mid,Z_mid = mid
    X_spiral,Y_spiral,Z_spiral = spiral
    
    # estimate projection of the lines in camera c
    xy_mid = np.vstack([F(np.vstack([X_mid,Y_mid,Z_mid]).T,ax),F(np.vstack([X_mid,Y_mid,Z_mid]).T,ay)]).T
    xy_spiral = np.vstack([F(np.vstack([X_spiral,Y_spiral,Z_spiral]).T,ax),F(np.vstack([X_spiral,Y_spiral,Z_spiral]).T,ay)]).T
    
    # make 3D figure
    axis = plt.figure(figsize=(10,8)).add_subplot(projection='3d')
    axis.set_xlabel('X'), axis.set_ylabel('Y'), axis.set_zlabel('Z')
    # plot box
    axis.plot([0,300],[0,0],[0,0],c='black'), axis.plot([0,0],[0,300],[0,0],c='black'), axis.plot([0,0],[0,0],[0,300],c='black')
    axis.plot([300,300],[0,300],[0,0],c='black'), axis.plot([300,300],[0,0],[0,300],c='black'), axis.plot([0,300],[300,300],[0,0],c='black')
    axis.plot([0,0],[300,300],[0,300],c='black'), axis.plot([0,300],[0,0],[300,300],c='black'), axis.plot([0,0],[0,300],[300,300],c='black')
    axis.plot([300,300],[300,300],[0,300],c='black'), axis.plot([300,300],[0,300],[300,300],c='black'), axis.plot([0,300],[300,300],[300,300],c='black')
    # plot torus
    axis.plot_surface(X, Y, Z, color='b', alpha=0.2)
    axis.plot(X_mid,Y_mid,Z_mid,'o-', color='red', linewidth=1)
    axis.plot(X_spiral,Y_spiral,Z_spiral,'o-', color='green', linewidth=1)
    plt.tight_layout(), plt.show()   
    
    # make projection of lines in cam c
    plt.figure()
    plt.title('reprojection on camera ' + str(c))
    plt.imshow(np.zeros([2160,2560]),cmap='gray')
    plt.plot(xy_mid[:,0],xy_mid[:,1],'o-',c='red')
    plt.plot(xy_spiral[:,0],xy_spiral[:,1],'o-',c='green')
    plt.show()
if __name__ == "__main__":
    main()