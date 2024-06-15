import numpy as np 

# camera IDs
cams = [0,1,2,3]

# load marker list for all cameras stored in a list
markers = [np.loadtxt('markers/markers_c'+str(cam)+'.txt') for cam in cams]
# get the 3D points of the markers, they are the same for every marker list
XYZ = markers[0][:,2:]
# get the cam points for each 3D marker dot as a set stored in a list
camPs = [ np.asarray([markers[0][i,:2],markers[1][i,:2],markers[2][i,:2],markers[3][i,:2]]) for i in range(len(markers[0]))]
# compute errors per case
for i in range(1,9):
    # get the calibration markers for each case
    ID = np.asarray(np.loadtxt('ID/ID'+str(i)+'.txt'),dtype=int)
    xyXYZ = [marker[ID] for marker in markers]
    
    ''' THIS IS YOUR PART - 1) INSERT YOUR CALIBRATION METHOD AND ESTIMATE THE CALIBRATION PARAMETERS FOR THE xyXYZ SET OF MARKER POINTS FOR EACH CASE '''
    # calibration
    from functions_proPTV import *
    from scipy.optimize import least_squares
    ax, ay = [], []
    for data in xyXYZ:
        def dFx(a):
            return F(data[:,2:],a) - data[:,0]    
        def dFy(a):
            return F(data[:,2:],a) - data[:,1]
        sx = least_squares(dFx,np.zeros(19),method='trf').x
        sy = least_squares(dFy,np.zeros(19),method='trf').x
        ax.append(sx)
        ay.append(sy)
    ''' END OF PART 1'''
    
    ''' THIS IS YOUR PART - 2) USE YOUR TRIANGULATION METHOD AND TRIANGULATE THE MARKER POINTS USING camPs STORED AS P=np.array([[X1,Y1,Z1],...]) '''
    # triangulate 3D position
    def NewtonSoloff_Triangulation(setP, ax, ay):
        setP = np.asarray(setP)
        foundSetPoints = np.argwhere(np.isnan(setP[:,0])==False)
        setP , aX , aY = setP[foundSetPoints[:,0]] , np.asarray(ax)[foundSetPoints[:,0]] , np.asarray(ay)[foundSetPoints[:,0]]
        P = np.array([150.0,150.0,150.0])
        for i in range(3):
            P += np.linalg.lstsq(Jacobian_Soloff(P, aX, aY),-Cost_Function(setP, P, aX, aY),rcond=None)[0]
        costsP = np.linalg.norm(Cost_Function(setP, P, aX, aY).reshape(len(aX),2),axis=1) # cost per cam
        return P, costsP
    P = np.asarray([NewtonSoloff_Triangulation(setP, ax, ay)[0] for setP in camPs])
    ''' END OF PART 2'''
    
    # save error
    error = XYZ-P
    np.savetxt('errors/error'+str(i)+'.txt',error)