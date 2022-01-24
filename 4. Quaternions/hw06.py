import sympy


# Task 6a
def proc(vector):
    X = sympy.Matrix([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])
    return X


def check_7112(theta1_half, theta2_half, v_theta, v_phi, u_theta, u_phi):
    s1 = sympy.sin(theta1_half)
    c1 = sympy.cos(theta1_half)
    s2 = sympy.sin(theta2_half)
    c2 = sympy.cos(theta2_half)

    v1 = sympy.cos(v_theta) * sympy.sin(v_phi)
    v2 = sympy.sin(v_theta) * sympy.sin(v_phi)
    v3 = sympy.cos(v_phi)
    u1 = sympy.cos(u_theta) * sympy.sin(u_phi)
    u2 = sympy.sin(u_theta) * sympy.sin(u_phi)
    u3 = sympy.cos(u_phi)

    v = sympy.Matrix([v1, v2, v3])  # column vectors
    u = sympy.Matrix([u1, u2, u3])

    # Maple script
    E = sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R1 = 2*(s1*v)*sympy.Transpose(s1*v) + (2*c1**2-1)*E + 2*c1*proc(s1*v)
    R2 = 2*(s2*u)*sympy.Transpose(s2*u) + (2*c2**2-1)*E + 2*c2*proc(s2*u)
    R21 = sympy.expand(R2*R1)
    c21 = c2*c1 - (sympy.Transpose(s2*u)*(s1*v)).row(0)[0]
    s21v21 = c2*s1*v + s2*c1*u + proc(s2*u)*(s1*v)
    RR21 = 2*s21v21*sympy.Transpose(s21v21) + (2*c21**2-1)*E + 2*c21*proc(s21v21)

    return sympy.simplify(sympy.expand(RR21-R21))


def R_from_theta_half_axis(theta_half, v):
    vx = sympy.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    # sin(theta)
    s = 2 * sympy.sin(theta_half) * sympy.cos(theta_half)
    c = 2*sympy.cos(theta_half)*sympy.cos(theta_half)-1
    # Identity matrix
    I = sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Rotation matrix
    R = I + (1-c)*vx*vx + s*vx
    return R


# Task 6b
def check_7116(theta1_half, theta2_half, v_theta, v_phi, u_theta, u_phi):
    s1 = sympy.sin(theta1_half)
    c1 = sympy.cos(theta1_half)
    s2 = sympy.sin(theta2_half)
    c2 = sympy.cos(theta2_half)

    v1 = sympy.cos(v_theta) * sympy.sin(v_phi)
    v2 = sympy.sin(v_theta) * sympy.sin(v_phi)
    v3 = sympy.cos(v_phi)
    u1 = sympy.cos(u_theta) * sympy.sin(u_phi)
    u2 = sympy.sin(u_theta) * sympy.sin(u_phi)
    u3 = sympy.cos(u_phi)

    v = sympy.Matrix([v1, v2, v3])
    u = sympy.Matrix([u1, u2, u3])

    q1 = sympy.Matrix([c1, s1*v])
    q2 = sympy.Matrix([c2, s2*u])

    c21 = c2 * c1 - (sympy.Transpose(s2 * u) * (s1 * v)).row(0)[0]
    s21v21 = c2 * s1 * v + s2 * c1 * u + proc(s2 * u) * (s1 * v)
    q21 = sympy.Matrix([c21, s21v21])

    qq21 = sympy.Matrix([[q2.row(0), -q2.row(1), -q2.row(2), -q2.row(3)],
                   [q2.row(1), q2.row(0), -q2.row(3), q2.row(2)],
                   [q2.row(2), q2.row(3), q2.row(0), -q2.row(1)],
                   [q2.row(3), -q2.row(2), q2.row(1), q2.row(0)]])*sympy.Matrix([q1.row(0), q1.row(1), q1.row(2), q1.row(3)])

    return qq21 - q21


def R_from_q(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    c_half_theta = q1
    c = 2*c_half_theta*c_half_theta-1  # finding cos(theta) from cos(theta/2)
    theta = sympy.acos(c)
    s = sympy.sin(theta)
    s_half_theta = sympy.sin(theta/2)
    v1 = q2/s_half_theta
    v2 = q3/s_half_theta
    v3 = q4/s_half_theta
    v = sympy.Matrix([v1, v2, v3])
    vx = sympy.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    I = sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Rotation matrix
    R = I + (1-c)*vx*vx + s*vx
    return R


def q_from_R(R):
    c = (sympy.trace(R)-1)/2
    theta = sympy.acos(c)
    c_half_theta = sympy.cos(theta/2)
    s_half_theta = sympy.sin(theta/2)
    s = sympy.sin(theta)
    v = 1/(2*s)*sympy.Matrix([R.row(2)[1] - R.row(1)[2], R.row(0)[2] - R.row(2)[0], R.row(1)[0] - R.row(0)[1]])
    q = sympy.Matrix([c_half_theta, s_half_theta*v.row(0)[0], s_half_theta*v.row(1)[0], s_half_theta*v.row(2)[0]])
    return q


# Task 6c
# 1. q1 and R1: rotation by pi/2 around x-axis
theta = sympy.pi/2
c = sympy.cos(theta)
s = sympy.sin(theta)
R1 = sympy.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
q1 = q_from_R(R1)
# 2. q2 and R2: rotation by pi/2 around y-axis
R2 = sympy.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
q2 = q_from_R(R2)
# 3. q21: using the composition of quaternions
q21 = sympy.Matrix([[q2.row(0), -q2.row(1), -q2.row(2), -q2.row(3)],
                   [q2.row(1), q2.row(0), -q2.row(3), q2.row(2)],
                   [q2.row(2), q2.row(3), q2.row(0), -q2.row(1)],
                   [q2.row(3), -q2.row(2), q2.row(1), q2.row(0)]])*sympy.Matrix([q1.row(0), q1.row(1), q1.row(2), q1.row(3)])

# 4. R21: R2*R1
R21 = R2*R1
# 5. R21: constructed from q21
RR21 = R_from_q(q21)
# 6. q21: constructed from R21
qq21 = q_from_R(R21)

# Verifying if R21&RR21 and q21&qq21 are the same
print(R21 == RR21)
print(q21 == qq21)

# Values for the dictionary
print(q1.evalf())
print(q2.evalf())
print(q21.evalf())
print(R1.evalf())
print(R2.evalf())
print(R21.evalf())

"""solution = {}
solution['q1'] = [[0.707106781186548], [0.707106781186548], [0], [0]]
solution['q2'] = [[0.707106781186548], [0], [0.707106781186548], [0]]
solution['q21'] = [[0.500000000000000], [0.500000000000000], [0.500000000000000], [-0.500000000000000]]
solution['R1'] = [[1.00000000000000, 0, 0], [0, 0, -1.00000000000000], [0, 1.00000000000000, 0]]
solution['R2'] = [[0, 0, 1.00000000000000], [0, 1.00000000000000, 0], [-1.00000000000000, 0, 0]]
solution['R21'] = [[0, 1.00000000000000, 0], [0, 0, -1.00000000000000], [-1.00000000000000, 0, 0]]

import json
with open("hw06.json", "w") as outfile:
    json.dump(solution, outfile)"""
