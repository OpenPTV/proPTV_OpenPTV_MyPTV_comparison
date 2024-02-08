import numpy as np
import matplotlib.pyplot as plt

from func import *

def main():
    # static parameter
    calibration_path = "calibration/c{cam}/soloff_c{cam}{xy}.txt"
    Vmin, Vmax = [0,0,0], [300,300,300]

    # user defined parameter
    c = 0 # camera ID
    xy = np.array([[1280,1080],[0,0],[2100,0]]) # image coordinates
    
    # plot parameter
    mu = 5 # line of sight:  LOF = P0 + mu*P1 , P0 - position vector , P1 - shift vector , mu - shift scalar
    
    # load calibration of camera c
    ax, ay = np.loadtxt(calibration_path.format(cam=c,xy="x"),delimiter=','), np.loadtxt(calibration_path.format(cam=c,xy="y"),delimiter=',')
    # calculate line of sight
    LOF = [proPTV_LineOfSight(p,c,Vmin,Vmax,ax,ay) for p in xy]
    # calculate cam position
    cam_position = Get_Closest_Point(LOF)
    print('estimated cam position: ' , cam_position)
    
    # plot line of sight
    axis = plt.figure(figsize=(10,8)).add_subplot(projection='3d')
    axis.set_xlabel('X'), axis.set_ylabel('Y'), axis.set_zlabel('Z')
    axis.plot([0,300],[0,0],[0,0],c='black'), axis.plot([0,0],[0,300],[0,0],c='black'), axis.plot([0,0],[0,0],[0,300],c='black')
    axis.plot([300,300],[0,300],[0,0],c='black'), axis.plot([300,300],[0,0],[0,300],c='black'), axis.plot([0,300],[300,300],[0,0],c='black')
    axis.plot([0,0],[300,300],[0,300],c='black'), axis.plot([0,300],[0,0],[300,300],c='black'), axis.plot([0,0],[0,300],[300,300],c='black')
    axis.plot([300,300],[300,300],[0,300],c='black'), axis.plot([300,300],[0,300],[300,300],c='black'), axis.plot([0,300],[300,300],[300,300],c='black')
    for lof in LOF:
        plt.plot( [lof[0,0],lof[0,0]+mu*lof[1,0]] , [lof[0,1],lof[0,1]+mu*lof[1,1]] , [lof[0,2],lof[0,2]+mu*lof[1,2]] ,'-',c='red')
    plt.plot(cam_position[0],cam_position[1],cam_position[2],'o',c='green')
    plt.tight_layout(), plt.show()    
if __name__ == "__main__":
    main()