import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R = np.array([[0.8047,   -0.5059,   -0.3106],
              [0.3106,    0.8047,   -0.5059],
              [0.5059,    0.3106,    0.8047]])
R_inv = np.linalg.inv(R)
o_0_beta_prime = np.array([[0], [0], [0]])
o_1_beta_prime = np.array([[1], [1], [1]])
I = np.identity(3)

# Computing o_0_prime_beta and o_1_prime_beta
o_0_prime_beta = np.dot(R, -o_0_beta_prime)
o_1_prime_beta = np.dot(R, -o_1_beta_prime)

# b1, b2 and b3 in β basis (is the standard basis)
b1_beta = np.array([[1], [0], [0]])
b2_beta = np.array([[0], [1], [0]])
b3_beta = np.array([[0], [0], [1]])

# Computing from β' to β
b1_pr_beta = [[R[0][0]], [R[1][0]], [R[2][0]]]
b2_pr_beta = [[R[0][1]], [R[1][1]], [R[2][1]]]
b3_pr_beta = [[R[0][2]], [R[1][2]], [R[2][2]]]

# Motion from Hw04
# Plotting vectors of β and β'
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(-2, 2, 1)
ax.set_ylim(-2, 2, 1)
ax.set_zlim(-2, 2, 1)

start_o = [0, 0, 0]
start_o_pr = o_1_prime_beta

# Plotting coordinate systems originating from points O and O'
ax.quiver(start_o[0], start_o[1], start_o[2], b1_beta[0][0], b1_pr_beta[1][0], b1_beta[2][0])
ax.quiver(start_o[0], start_o[1], start_o[2], b2_beta[0][0], b2_pr_beta[1][0], b2_beta[2][0])
ax.quiver(start_o[0], start_o[1], start_o[2], b3_beta[0][0], b3_pr_beta[1][0], b3_beta[2][0])
ax.quiver(start_o_pr[0][0], start_o_pr[1][0], start_o_pr[2][0], b1_pr_beta[0][0], b1_pr_beta[1][0], b1_pr_beta[2][0], color="g")
ax.quiver(start_o_pr[0][0], start_o_pr[1][0], start_o_pr[2][0], b2_pr_beta[0][0], b2_pr_beta[1][0], b2_pr_beta[2][0], color="g")
ax.quiver(start_o_pr[0][0], start_o_pr[1][0], start_o_pr[2][0], b3_pr_beta[0][0], b3_pr_beta[1][0], b3_pr_beta[2][0], color="g")

# Finding R-I matrix and its rank
ri = R-I
rank = np.linalg.matrix_rank(ri)
# Left side of the equation
ls = np.dot(ri, ri)
# Right side of the equation
rs_0 = np.dot(-ri, o_0_prime_beta)  # the right side of the equation will be 0 for a_0
rs_1 = np.dot(-ri, o_1_prime_beta)

# Computing a_0
# We are in the case where rank(R-I) = 2 and o_prime_beta is 0
# Therefore, the equation will have a one-dimensional space of solutions.
# The null space and range(R-I) will intersect only in the zero vector
a_0 = null_space(ls)
a_0_dir = a_0
a_0_point = np.array([[0], [0], [0]])
x0_values = [a_0_dir[0][0], a_0_point[0][0]]
y0_values = [a_0_dir[1][0], a_0_point[1][0]]
z0_values = [a_0_dir[2][0], a_0_point[2][0]]
ax.plot(x0_values, y0_values, z0_values, color="m")

# Computing a_1
# We are in the case where rank(R-I) = 2 and o_prime_beta is different from 0
# o_prime_beta is in the span of R-I
a_1_point = np.linalg.pinv(ri).dot(-o_1_prime_beta)
a_1_point1 = a_1_point
a_1_point2 = a_0_dir + a_1_point1
a_1_dir = a_0_dir
x1_values = [a_1_point1[0][0], a_1_point2[0][0]]
y1_values = [a_1_point1[1][0], a_1_point2[1][0]]
z1_values = [a_1_point1[2][0], a_1_point2[2][0]]
ax.plot(x1_values, y1_values, z1_values, color="b")

# Computing the axis of rotation
# We know that R*x = x => (R-I)*x = 0
r = null_space(ri)
# r coincides with a_0 and is parallel to a_1
r_dir = r
r_point = np.array([[0], [0], [0]])
xr_values = [r_dir[0][0], r_point[0][0]]
yr_values = [r_dir[1][0], r_point[1][0]]
zr_values = [r_dir[2][0], r_point[2][0]]
ax.plot(xr_values, yr_values, zr_values, color="k")

# Computing the generators of sigma plane
sigma1 = o_1_prime_beta
sigma2 = np.cross(sigma1.T, r.T).T
ax.quiver(0, 0, 0, sigma1[0][0], sigma1[1][0], sigma1[2][0], color="y")
ax.quiver(0, 0, 0, sigma2[0][0], sigma2[1][0], sigma2[2][0], color="c")

# Computing point p, p', p''
p = a_1_point
p_pr = np.dot(R, p)
p_pr_pr = p_pr + o_1_prime_beta
ax.scatter(p[0][0], p[1][0], p[2][0], color="k")
ax.scatter(p_pr[0][0], p_pr[1][0], p_pr[2][0], color="g")
ax.scatter(p_pr_pr[0][0], p_pr_pr[1][0], p_pr_pr[2][0], color="y")
plt.show()

# Converting the solutions to list for the json file
a_0_dir = a_0_dir.tolist()
a_0_point = a_0_point.tolist()
a_1_dir = a_1_dir.tolist()
a_1_point = a_1_point.tolist()
sigma1 = sigma1.tolist()
sigma2 = sigma2.tolist()
p = p.tolist()
p_pr = p_pr.tolist()
p_pr_pr = p_pr_pr.tolist()

# json file
"""solution = {"a_0_dir": a_0_dir, "a_0_point": a_0_point, "a_1_dir": a_1_dir, "a_1_point": a_1_point, "sigma1": sigma1,
            "sigma2": sigma2, "p": p, "p_pr": p_pr, "p_pr_pr": p_pr_pr}
import json
with open("hw05.json", "w") as outfile:
    json.dump(solution, outfile)"""