'''
    NOTES: 
    # convention of marker sorting: left to right (x) then top to bottom (y)
    # note that y top denotes to the smallest y because in the image as the y axis is inversed
    # Therefore, in the images the top left point is (0,0,0) in the first frame
    
    TODO: 
    # refine calibration and reduce calibration error by using the exact 40 mm constraint of the distance between marker points
    # OR take the first plane and correct all other planes by straight rays of light inside the volume of interest
'''

# Load libaries
import cv2, sys, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define code parameter here
# %%
class Parameter:
    cams, planes = [0,1,2,3], [1,2,3,4,5,6,12,13,14,15,16,17,18,19,20,21,22,23]
    
    marker_distance, Z0 = (40,40), 0 # X[mm], Y[mm], Z[mm]
    marker_size = (25, 19)  # x, y
    image_size = (2560, 2048)  # x[px], y[px]
    
    Zeros = 2
    markerList = "Ilmenau_xy/c{cam}/marker/c{cam}_{plane}.txt" 
    markerImage = "Ilmenau_xy/c{cam}/c{cam}_{plane}_01.tif"
    markerOutput = "c{cam}_xyXYZ.txt"
# %%

def DLT(P1, P2, P3, P4, xy1, xy2, xy3, xy4):
    """
    Args:
        P1, P2, P3, P4: 3x4 projection matrices for cameras i in (1, 2, 3, 4) so that: xy_i = P_i * XYZ and XYZ=(X,Y,Z,1)
        xy1, xy2, xy3, xy4: 2D image points (x, y) for each camera
    
    Returns:
        3D point (X, Y, Z) in world coordinates
    Meaning:
        Perform DLT with 4 cameras to reconstruct a 3D point with projection matrices of cameras 2,3,4: P2,P3,P4, relative to camera 1.
        The system solves the homogeneous system A * XYZ = 0.
        From a pinhole model we know that xy_i = P_i * XYZ with xy_i = [x',y',w] and x=x'/w and y=y'/w building 2 equations for each camera: 
                x_i = (P_i*XYZ)_row1 / (P_i*XYZ)_raw3  -> (x_i*P_i_raw1 - P_i_raw3) * XYZ = 0
                y_i = (P_i*XYZ)_row2 / (P_i*XYZ)_raw3  -> (y_i*P_i_raw2 - P_i_raw3) * XYZ = 0
        This we can build matrix the matrix equation A * XYZ = 0 using all four cameras and the two equations above for each of them.
        Due to noise we do not have A*XYZ=0 instead we have A*XYZ\approx 0 with an non-trivial solution XYZ!=0.
        To find the correct solution we use SVD
        SVD decomposes A = U S V^T, where S is diagonal with singular values.
        The solution we need is the singular vector V_T which corresponding to the smallest singular value (i.e. the last one in numpys output)
        This value minimizes A*XYZ.
    """
    # Construct the 8x4 matrix A using equations from all 4 cameras
    A = np.vstack([
        xy1[1] * P1[2, :] - P1[1, :],  # y1 * P1_3 - P1_2
        P1[0, :] - xy1[0] * P1[2, :],  # P1_1 - x1 * P1_3
        xy2[1] * P2[2, :] - P2[1, :],  # y2 * P2_3 - P2_2
        P2[0, :] - xy2[0] * P2[2, :],  # P2_1 - x2 * P2_3
        xy3[1] * P3[2, :] - P3[1, :],  # y3 * P3_3 - P3_2
        P3[0, :] - xy3[0] * P3[2, :],  # P3_1 - x3 * P3_3
        xy4[1] * P4[2, :] - P4[1, :],  # y4 * P4_3 - P4_2
        P4[0, :] - xy4[0] * P4[2, :]]) # P4_1 - x4 * P4_3
    # Solve using Singular Value Decomposition (SVD)
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    # The corrct 3D point is the last column of Vh 
    XYZ = Vh[-1, 0:3] / Vh[-1, 3]  # Normalize by the homogeneous coordinate
    return XYZ

def main():
    # load parameters 
    params = Parameter()    
    
    # define first plane XYZ as reference with Z = 0
    X, Y, Z = np.meshgrid(np.arange(0,params.marker_size[0]*params.marker_distance[0],params.marker_distance[0]),
                          -np.arange(0,params.marker_size[1]*params.marker_distance[1],params.marker_distance[1]),
                          np.linspace(params.Z0,params.Z0,1))
    # initalize a XYZ plane for each plane position equal to the first plane with Z=0
    # we dont know the exact 3D XYZ position of the planes yet
    XYZ = [np.asarray(np.vstack([X.ravel(),Y.ravel(),Z.ravel()]).T, dtype=np.float32) for plane in params.planes]
    
    # calibrate cameras individually
    xy_c, P_c, R_c, ret_c, M_c, d_c, r_c, t_c, pos_c = [], [], [], [], [], [], [], [], []
    for i,c in enumerate(params.cams):
        xy = [np.asarray(np.loadtxt(params.markerList.format(cam=c,plane=str(t).zfill(params.Zeros)),skiprows=1), dtype=np.float32) for t in params.planes]
        xy_c.append(xy)
        ret, M, d, r, t = cv2.calibrateCamera(XYZ,xy,params.image_size,None,None) 
        ret_c.append(ret), M_c.append(M), d_c.append(d), r_c.append(r), t_c.append(t)
        # estimate rotation matrix
        R = cv2.Rodrigues(r[0])[0] 
        R_c.append(R)
        # estimate camera position
        pos = -np.dot(R.T, t[0]).ravel()
        pos_c.append(pos)
        # build projection matizes relative to camera 0 
        if i==0: # projection matrix for camera 0
            RT = np.concatenate([R, t[0]], axis=-1)
            P_c.append(M @ RT)
        else: # projection matrix relative to camera 0 for camera 1, 2 and 3
            # stereo matching of the FIRST 3D marker plane - use only cam 0 and cam i with i in (1,2,3)
            # the assumption of straight lines for light rays inside the medium is needed here because only the first plane 3D positions are known
            ret, CM0, dist0, CM1, dist1, R, T, E, F = cv2.stereoCalibrate(XYZ[:1], xy_c[0][:1], xy_c[i][:1], M_c[0], d_c[0], M_c[i], d_c[i], params.image_size)
            # projection matrix for camera i with respect to camera 0
            RT = np.concatenate([R@R_c[0], (R@t_c[0][0]+T)], axis = -1)
            P_c.append(M @ RT)
    # compute 3D marker positions using all cameras and the DLT algorithm
    P = []
    for p in tqdm(range(len(params.planes)),desc='DLT'):
        # For each plane, reconstruct 3D points using corresponding 2D points from all 4 cameras
        markers_p = []
        for xy0, xy1, xy2, xy3 in zip(xy_c[0][p], xy_c[1][p], xy_c[2][p], xy_c[3][p]):
            markers_p.append( DLT(P_c[0], P_c[1], P_c[2], P_c[3], xy0, xy1, xy2, xy3) )
        P.append(np.asarray(markers_p,dtype=np.float32))
    P[0] = XYZ[0]
    # recalibrate cameras with the reconstructed XYZ points of all plates
    for i,c in enumerate(params.cams):
        ret, M, d, r, t = cv2.calibrateCamera(P,xy_c[i],params.image_size,M_c[i],d_c[i],flags=cv2.CALIB_USE_INTRINSIC_GUESS) 
        ret_c[i], M_c[i], d_c[i], r_c[i], t_c[i] = ret, M, d, r, t
        R = cv2.Rodrigues(r[0])[0] 
        R_c[i] = R
        pos = -np.dot(R.T, t[0]).ravel()
        pos_c[i] = pos
        print('position cam '+str(c)+': ', pos)
         
    # save out Soloff dataset which is the xyXYZ list for each camera containing all planes
    for c in range(len(params.cams)):
        xyXYZ = np.concatenate([np.append(xy_c[c][i],P[i],axis=1) for i in range(len(params.planes))])
        np.savetxt(params.markerOutput.format(cam=str(params.cams[c])), xyXYZ, header='x,y,X,Y,Z')
    
    # verify code and calculate some error metrics
    '''HERE Chose which camera and which plane the metrics are calculated and plotted for'''
    c = 0 # for c in range(len(params.cams)):
    i = 6 # for i in range(6,len(params.planes)):
    print('')
    print('2D projection error')
    # project xy = P*XYZ on camera i
    p, _ = cv2.projectPoints(P[i], r_c[c][0], t_c[c][0], M_c[c], d_c[c])
    p = p.reshape(int(params.marker_size[0]*params.marker_size[1]),2)
    # 2D error calculation
    error_xy = np.linalg.norm(xy_c[c][i]-p,axis=1)
    print(' Camera '+str(c)+' - Plane '+str(i)) 
    print('  Mean: '+str(np.round(np.mean(error_xy),2)), ' Max: '+str(np.round(np.max(error_xy),2)), ' STD: '+str(np.round(np.std(error_xy),2)))
    # plot 2D reprojection
    plt.figure()
    plt.imshow(cv2.imread(params.markerImage.format(cam=c,plane=str(params.planes[i]).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED),cmap='gray')
    plt.plot(xy_c[c][i][:,0],xy_c[c][i][:,1],'o',c='green',label='marker detection')
    plt.plot(p[:,0],p[:,1],'o',c='red',label='reprojection')
    plt.legend()
    plt.show()
    print('')
    print('3D plane errors')
    horizontal_error_XYZ = np.concatenate([np.diff(P[i][n:params.marker_size[1],0]) for n in range(params.marker_size[0])])
    vertical_error_XYZ = np.concatenate([np.diff(P[i][n::params.marker_size[0],1]) for n in range(params.marker_size[1])])
    print(' Plane '+str(i)) 
    print('  Horizontal - Mean: '+str(np.round(np.mean(horizontal_error_XYZ),2)), ' STD: '+str(np.round(np.std(horizontal_error_XYZ),2)))
    print('  Vertical - Mean: '+str(np.round(np.mean(vertical_error_XYZ),2)), ' STD: '+str(np.round(np.std(vertical_error_XYZ),2)))
    # plot 3D positions, take attention on the coordinate system orientation
    fig = plt.figure(figsize=(12,12))
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlabel('Z [mm]'), axis.set_ylabel('X [mm]'), axis.set_zlabel('Y [mm]')
    axis.set_xlim(-4000,4000), axis.set_ylim(-4000,4000), axis.set_zlim(-1500,880) # 2.38 x 7.0 m , cameras about 20cm away from the bottom and top plate
    axis.scatter(pos_c[0][2],pos_c[0][0],pos_c[0][1],label='c0',c='blue')  
    axis.scatter(pos_c[1][2],pos_c[1][0],pos_c[1][1],label='c1',c='green')  
    axis.scatter(pos_c[2][2],pos_c[2][0],pos_c[2][1],label='c2',c='brown')  
    axis.scatter(pos_c[3][2],pos_c[3][0],pos_c[3][1],label='c3',c='orange')  
    axis.scatter(XYZ[0][:,2],XYZ[0][:,0],XYZ[0][:,1],c='red')   
    [axis.scatter(P[i][:,2],P[i][:,0],P[i][:,1],c='black') for i in range(1,len(params.planes))]
    # plot geometry
    theta = np.linspace(0, 2*np.pi, 100)
    geometry_down = [3500*np.cos(theta),3500*np.sin(theta)+480,np.zeros_like(theta)-1500]
    geometry_up = [3500*np.cos(theta),3500*np.sin(theta)+480,np.zeros_like(theta)+880]
    axis.plot(geometry_down[0],geometry_down[1],geometry_down[2],c='black')
    axis.plot(geometry_up[0],geometry_up[1],geometry_up[2],c='black')
    axis.plot(0,480,-1500,'x',c='black')
    axis.plot(0,480,880,'x',c='black')
    plt.legend()
    plt.show()
if __name__ == "__main__":

    main()

