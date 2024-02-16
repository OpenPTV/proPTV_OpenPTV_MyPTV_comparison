import numpy as np

def Rotate(point, angle, axis):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return np.dot(rotation_matrix, point)
 
def Torus(R,r,N,loops,offset,angle,axis):
    # Define the torus with major and minor radii and rotate to desired axis
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, N), np.linspace(0, 2*np.pi, N))
    X = (R + r*np.cos(phi)) * np.cos(theta)
    Y = (R + r*np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)
    for i in range(len(X)):
        for j in range(len(X[0])):
            x, y, z = X[i, j], Y[i, j], Z[i, j]
            X[i, j], Y[i, j], Z[i, j] = Rotate([x, y, z], angle, axis)
    # Define middle line and rotate to desired axis
    t_mid = np.linspace(0, 2*np.pi, N)
    X_mid = R * np.cos(t_mid)
    Y_mid = R * np.sin(t_mid)
    Z_mid = np.zeros_like(t_mid)
    for i in range(len(X_mid)):
        X_mid[i], Y_mid[i], Z_mid[i] = Rotate([X_mid[i], Y_mid[i], Z_mid[i]], angle, axis)
    # Define spiral line and rotate to desired axis
    t_spiral = np.linspace(0, 2*np.pi * loops, N)
    X_spiral = (R + r*np.cos(t_spiral)) * np.cos(t_spiral/loops)
    Y_spiral = (R + r*np.cos(t_spiral)) * np.sin(t_spiral/loops)
    Z_spiral = r * np.sin(t_spiral)
    for i in range(len(X_spiral)):
        X_spiral[i], Y_spiral[i], Z_spiral[i] = Rotate([X_spiral[i], Y_spiral[i], Z_spiral[i]], angle, axis)
    return [X+offset,Y+offset,Z+offset], [X_mid+offset,Y_mid+offset,Z_mid+offset], [X_spiral+offset,Y_spiral+offset,Z_spiral+offset]