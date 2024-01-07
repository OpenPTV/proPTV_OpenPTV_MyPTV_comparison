import numpy as np

def F(XYZ, a):
    '''
        soloff polynom
    '''
    X , Y , Z = XYZ[0] , XYZ[1] , XYZ[2]
    return ( a[0] 
                + X * ( X*(a[9]*X+a[11]*Y+a[14]*Z+a[4]) + a[13]*Y*Z + a[6]*Y + a[7]*Z + a[1] ) 
                + Y * ( Y*(a[12]*X+a[10]*Y+a[15]*Z+a[5]) + a[8]*Z + a[2] ) 
                + Z * ( Z*(a[17]*X+a[18]*Y+a[16]) + a[3] ) ) 

def dFdx(XYZ, a):
    '''
        derivative of soloff polynom by x
    '''
    X , Y , Z = XYZ[0] , XYZ[1] , XYZ[2]
    return (3 * a[9] * pow(X, 2) + 2 * a[11] * X * Y + 2 * a[14] * X * Z + 2 * a[4] * X 
            + a[12] * pow(Y, 2) + a[13] * Y * Z + a[6] * Y
            + a[17] * pow(Z, 2) + a[7] * Z + a[1])
            
def dFdy(XYZ, a):
    '''
        derivative of soloff polynom by y
    '''
    X , Y , Z = XYZ[0] , XYZ[1] , XYZ[2]
    return (a[11] * pow(X, 2) + 2 * a[12] * X * Y + a[13] * X * Z + a[6] * X 
            + 3 * a[10] * pow(Y, 2) + 2 * a[15] * Y * Z + 2 * a[5] * Y 
            + a[18] * pow(Z, 2) + a[8] * Z + a[2])
            
def dFdz(XYZ, a):
    '''
        derivative of soloff polynom by z
    '''
    X , Y , Z = XYZ[0] , XYZ[1] , XYZ[2]
    return  (2 * Z * (a[17] * X + a[18] * Y + a[16]) 
            + X * (a[14] * X + a[13] * Y + a[7]) 
            + Y * (a[15] * Y + a[8]) + a[3])

def Cost_Function(setP,P,ax,ay):
    '''
        calculates the cost function per active camera
        (F(P) - camP) for each active cam
        difference between particle camera position and reprojected camera position
    '''
    cost = np.ravel( [[F(P,ax[i])-setP[i][0] , F(P,ay[i])-setP[i][1]] for i in np.arange(len(ax))] )
    return cost

def Jacobian_Soloff(P,ax,ay):
    '''
        calculates the Jacobian matrix of the soloff polynom for gradient descent algorithm
    '''
    jac = [ [[dFdx(P,ax[i]) , dFdy(P,ax[i]) , dFdz(P,ax[i])] , [dFdx(P,ay[i]) , dFdy(P,ay[i]) , dFdz(P,ay[i])]] for i in np.arange(len(ax)) ]
    J = np.asarray(sum(jac,[])) # [j for i in jac for j in i]
    return J

def proPTV_LineOfSight(p,c,Vmin,Vmax,ax,ay):
    '''
        calculate line of sight
    '''
    P1 = np.array([ (Vmax[0]+Vmin[0])/2, Vmin[1], (Vmax[2]+Vmin[2])/2 ])
    P2 = np.array([ (Vmax[0]+Vmin[0])/2, Vmax[1], (Vmax[2]+Vmin[2])/2 ])
    for n in range(5):
        P1 += np.linalg.lstsq(Jacobian_Soloff(P1,[ax],[ay]),-np.array([F(P1,ax)-p[0], F(P1,ay)-p[1]]),rcond=None)[0] 
        P2 += np.linalg.lstsq(Jacobian_Soloff(P2,[ax],[ay]),-np.array([F(P2,ax)-p[0], F(P2,ay)-p[1]]),rcond=None)[0] 
    return np.array([ P2 , P1-P2 ])

def dist_Point_Line(p,P,V):
    u = np.cross(V,P-p)
    return np.dot( u , u )
def Grad_dist_Point_Line(p,P,V):
    V = V / np.linalg.norm(V)
    return (2*V*(np.dot(V,(P-p)))) - 2*(np.dot(V,V))*(P-p)
def Get_Closest_Point(lines):
    '''
    estimate clostest point of multiple lines
    '''
    p = np.zeros(3)
    for i in range(1000):
        p -= 0.1 * np.sum([Grad_dist_Point_Line(p, line[0], line[1]) for line in lines],axis=0)
    return p