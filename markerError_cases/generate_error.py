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
    ax, ay = 
    ''' END OF PART 1'''
    
    ''' THIS IS YOUR PART - 2) USE YOUR TRIANGULATION METHOD AND TRIANGULATE THE MARKER POINTS USING camPs STORED AS P=np.array([[X1,Y1,Z1],...]) '''
    # triangulate 3D position
    P = 
    ''' END OF PART 2'''
    
    # save error
    error = XYZ-P
    np.savetxt('errors/error'+str(i)+'.txt',error)