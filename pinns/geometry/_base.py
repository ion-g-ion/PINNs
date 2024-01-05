import numpy as np

def rotation_matrix_3d(rotation):
    # Extract the angles from the rotation tuple
    alpha, beta, gamma = rotation

    # Compute the sine and cosine values of the angles
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)

    # Create the rotation matrix
    rotation_matrix = np.array([[cb*cg, sa*sb*cg-ca*sg, ca*sb*cg+sa*sg],
                                [cb*sg, sa*sb*sg + ca*cg, ca*sb*sg-sa*cg],
                                [-sb, sa*cb, ca*cb]])

    return rotation_matrix