import numpy as np


def fkt(mechanism, joints):
    # Creating empty lists
    theta_offset = []
    d = []
    a = []
    alpha = []
    theta = []
    # Creating the identity matrix in order to not influence the first product (g = g * A01)
    g = np.identity(4)
    # Assigning the values of the dictionary to lists (they can be access more easily)
    for index in range(0, 6):
        theta_offset.append(mechanism[f'theta{index + 1} offset'])
        d.append(mechanism[f'd{index + 1}'])
        a.append(mechanism[f'a{index + 1}'])
        alpha.append(mechanism[f'alpha{index + 1}'])
        theta.append(joints[f'theta{index + 1}'])
    # Computing the transformation matrix G = A10*A21*A32*A43*A54*A65
    for i in range(0, 6):
        g = np.dot(g, np.array([[np.cos(theta_offset[i] + theta[i]), -np.sin(theta_offset[i] + theta[i]) * np.cos(alpha[i]),
                        np.sin(theta_offset[i] + theta[i]) * np.sin(alpha[i]),
                        a[i] * np.cos(theta_offset[i] + theta[i])],
                       [np.sin(theta_offset[i] + theta[i]), np.cos(theta_offset[i] + theta[i]) * np.cos(alpha[i]),
                        -np.cos(theta_offset[i] + theta[i]) * np.sin(alpha[i]),
                        a[i] * np.sin(theta_offset[i] + theta[i])],
                       [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
                       [0, 0, 0, 1]]))
    # Extracting the rotation matrix from G
    rot = np.array([g[0:3, 0:3]])
    # Extracting the translation vector from G
    tr = np.array([g[0:3, 3]])
    # Creating the dictionary
    dict = {'r': rot, 't': tr.T}
    return dict