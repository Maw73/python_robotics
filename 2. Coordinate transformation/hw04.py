import numpy as np
import matplotlib.pyplot as plt

# approximate rotation
# R has the β' vectors expressed in β basis
R = np.array([[0.8047,   -0.5059,   -0.3106],
              [0.3106,    0.8047,   -0.5059],
              [0.5059,    0.3106,    0.8047]])
R_inv = np.linalg.inv(R)
# translation O'O_β'
o_beta_pr = np.array([[1], [1], [1]])
# x_β
x_beta = np.array([[1], [2], [3]])
# Computing x'_β' using Equation 4.5
x_pr_beta_pr = np.dot(R, x_beta) + o_beta_pr
# b1, b2 and b3 in β basis (is the standard basis)
b1_beta = np.array([[1], [0], [0]])
b2_beta = np.array([[0], [1], [0]])
b3_beta = np.array([[0], [0], [1]])

# Computing from β to β'
b1_beta_pr = [[R_inv[0][0]], [R_inv[1][0]], [R_inv[2][0]]]
b2_beta_pr = [[R_inv[0][1]], [R_inv[1][1]], [R_inv[2][1]]]
b3_beta_pr = [[R_inv[0][2]], [R_inv[1][2]], [R_inv[2][2]]]
# Computing from β' to β
b1_pr_beta = [[R[0][0]], [R[1][0]], [R[2][0]]]
b2_pr_beta = [[R[0][1]], [R[1][1]], [R[2][1]]]
b3_pr_beta = [[R[0][2]], [R[1][2]], [R[2][2]]]

# Computing OZ
oz_beta = (np.dot(R, x_beta) - np.dot(R, o_beta_pr)).tolist()

# Plotting vectors of β amd β'
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(-2, 3, 1)
ax.set_ylim(-2, 4, 1)
ax.set_zlim(-2, 5, 1)

start_o = [0, 0, 0]
start_o_pr = o_beta_pr

# Plotting vectors in standard basis
"""ax.quiver(start_o[0], start_o[1], start_o[2], b1_beta[0][0], b1_pr_beta[1][0], b1_beta[2][0])
ax.quiver(start_o[0], start_o[1], start_o[2], b2_beta[0][0], b2_pr_beta[1][0], b2_beta[2][0])
ax.quiver(start_o[0], start_o[1], start_o[2], b3_beta[0][0], b3_pr_beta[1][0], b3_beta[2][0])
ax.quiver(start_o[0], start_o[1], start_o[2], b1_pr_beta[0][0], b1_pr_beta[1][0], b1_pr_beta[2][0], color="g")
ax.quiver(start_o[0], start_o[1], start_o[2], b2_pr_beta[0][0], b2_pr_beta[1][0], b2_pr_beta[2][0], color="g")
ax.quiver(start_o[0], start_o[1], start_o[2], b3_pr_beta[0][0], b3_pr_beta[1][0], b3_pr_beta[2][0], color="g")"""

# Plotting coordinate systems originating from points O and O'
ax.quiver(start_o[0], start_o[1], start_o[2], b1_beta[0][0], b1_pr_beta[1][0], b1_beta[2][0])
ax.quiver(start_o[0], start_o[1], start_o[2], b2_beta[0][0], b2_pr_beta[1][0], b2_beta[2][0])
ax.quiver(start_o[0], start_o[1], start_o[2], b3_beta[0][0], b3_pr_beta[1][0], b3_beta[2][0])
ax.quiver(start_o_pr[0][0], start_o_pr[1][0], start_o_pr[2][0], b1_pr_beta[0][0], b1_pr_beta[1][0], b1_pr_beta[2][0], color="g")
ax.quiver(start_o_pr[0][0], start_o_pr[1][0], start_o_pr[2][0], b2_pr_beta[0][0], b2_pr_beta[1][0], b2_pr_beta[2][0], color="g")
ax.quiver(start_o_pr[0][0], start_o_pr[1][0], start_o_pr[2][0], b3_pr_beta[0][0], b3_pr_beta[1][0], b3_pr_beta[2][0], color="g")
# Plotting x in beta
ax.quiver(start_o[0], start_o[1], start_o[2], x_beta[0][0], x_beta[1][0], x_beta[2][0], color="m")
# Plotting y in beta'
ax.quiver(start_o_pr[0][0], start_o_pr[1][0], start_o_pr[2][0], x_beta[0][0], x_beta[1][0], x_beta[2][0], color="r")
# Plotting OZ in beta
ax.quiver(start_o[0], start_o[1], start_o[2], oz_beta[0][0], oz_beta[1][0], oz_beta[2][0], color="k")
plt.show()

# json file
"""solution = {"b1_beta_pr": b1_beta_pr, "b2_beta_pr": b2_beta_pr, "b3_beta_pr": b3_beta_pr, "b1_pr_beta": b1_pr_beta,
            "b2_pr_beta": b2_pr_beta, "b3_pr_beta": b3_pr_beta, "oz_beta": oz_beta}

import json
with open("hw04.json", "w") as outfile:
    json.dump(solution, outfile)"""
