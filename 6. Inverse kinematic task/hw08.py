import math
import os
import sympy as sp
from sympy import symbols
from sympy.core.sympify import sympify
from sympy.polys.polytools import Poly
from fractions import Fraction
from math import modf
import numpy as np


def get_power(num):
    string = format(num, '.1e')
    sign = ((string.split('e'))[1])[0]
    number = (string.split(sign))[1]
    if number[0] == '0':
        number = number[1]
    power = int(sign + number)
    return power


def rat_approx(n, tol):
    if tol >= 1:
        f = Fraction(int(modf(n)[1]), 1)
    if tol < 1:
        e = get_power(tol)
        f = Fraction(int(modf(n * 10 ** (-e))[1]), int(10 ** (-e)))
    return f


def exact_cs(angle, tol):
    if abs(angle - math.pi) < tol or abs(angle + math.pi) < tol:
        c = -1
        s = 0
    t = rat_approx(math.tan(angle / 2), tol / 10)
    c = Fraction(1 - t ** 2, 1 + t ** 2)
    s = Fraction(2 * t, 1 + t ** 2)
    return [c, s]


def q2r(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    R = (1/(q1**2 + q2**2 + q3**2 + q4**2)) * np.array([[q1 * q1 + q2 * q2 - q3 * q3 - q4 * q4, 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
                  [2 * (q2 * q3 + q1 * q4), q1 * q1 - q2 * q2 + q3 * q3 - q4 * q4, 2 * (q3 * q4 - q1 * q2)],
                  [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4]])
    return R



def exact_rot(q, tol):
    tol_q = tol
    while True:
        q_rat = np.array([0, 0, 0, 0])
        q_rat = q_rat.astype('object')
        for i in range(4):
            q_rat[i] = rat_approx(q[i], tol_q)
        R = q2r(q_rat)
        if np.linalg.norm(R - q2r(q)) < tol:
            return R
        else:
            tol_q = tol_q/10


def rational_mechanism(mechanism, tol):
    my_list = []
    dict = {}
    for index in range(1, 7):
        dict[f'theta{index} offset'] = mechanism[f'theta{index} offset']
    for index in range(1, 7):
        dict[f'd{index}'] = rat_approx(float(mechanism[f'd{index}']), tol)
    for index in range(1, 7):
        dict[f'a{index}'] = rat_approx(float(mechanism[f'a{index}']), tol)
    for index in range(1, 7):
        my_list += exact_cs(float(mechanism[f'alpha{index}']), tol)
    for index in range(1, 7):
        dict[f'cos alpha{index}'] = exact_cs(float(mechanism[f'alpha{index}']), tol)[0]
        dict[f'sin alpha{index}'] = exact_cs(float(mechanism[f'alpha{index}']), tol)[1]
    return dict


def rational_pose(pose, tol):
    tr = np.array([0, 0, 0])
    last_row = np.array([0, 0, 0, 1])
    last_row = last_row.astype('object')
    for i in range(0, 4):
        last_row[i] = Fraction(last_row[i])
    R = exact_rot(pose['q'], tol)
    tr = tr.astype('object')
    for i in range(0, 3):
        tr[i] = rat_approx((pose['t'])[i], tol)
    # Appending the matrix and the vector
    matr_temp = np.c_[R, tr]
    matr = np.vstack([matr_temp, last_row])
    return matr


def ikt_eqs(mechanism, pose, tol):
    # Defining sympy variables for sine and cos of theta + theta_offset
    c1, c2, c3, c4, c5, c6, s1, s2, s3, s4, s5, s6 = symbols("c1, c2, c3, c4, c5, c6, s1, s2, s3, s4, s5, s6")
    list1 = []
    list2 = []
    M1 = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    M2 = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    a = []
    d = []
    cos_alpha = []
    sin_alpha = []
    # Computing the matrix of the pose of the end effector
    pose_matrix = rational_pose(pose, tol)
    pose_matrix = sp.Matrix(pose_matrix)
    # Creating lists for a, d, cos and sin alpha
    mech = rational_mechanism(mechanism, tol)
    for i in range(6):
        a.append(mech[f'a{i+1}'])
        d.append(mech[f'd{i+1}'])
        cos_alpha.append(mech[f'cos alpha{i+1}'])
        sin_alpha.append(mech[f'sin alpha{i+1}'])
    # Computing the equations for the pose of the end effector
    # M_i^(i-1) for i=4,5,6
    M11 = sp.Matrix([[c4, -s4, 0, 0], [s4, c4, 0, 0], [0, 0, 1, d[3]], [0, 0, 0, 1]]) * sp.Matrix(([[1, 0, 0, a[3]], [0, cos_alpha[3], -sin_alpha[3], 0], [0, sin_alpha[3], cos_alpha[3], 0], [0, 0, 0, 1]]))
    M12 = sp.Matrix([[c5, -s5, 0, 0], [s5, c5, 0, 0], [0, 0, 1, d[4]], [0, 0, 0, 1]]) * sp.Matrix(([[1, 0, 0, a[4]], [0, cos_alpha[4], -sin_alpha[4], 0], [0, sin_alpha[4], cos_alpha[4], 0], [0, 0, 0, 1]]))
    M13 = sp.Matrix([[c6, -s6, 0, 0], [s6, c6, 0, 0], [0, 0, 1, d[5]], [0, 0, 0, 1]]) * sp.Matrix(([[1, 0, 0, a[5]], [0, cos_alpha[5], -sin_alpha[5], 0], [0, sin_alpha[5], cos_alpha[5], 0], [0, 0, 0, 1]]))
    # M_i^(i-1) for i=1,2,3
    M21 = sp.Matrix([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, 1, d[0]], [0, 0, 0, 1]]) * sp.Matrix(([[1, 0, 0, a[0]], [0, cos_alpha[0], -sin_alpha[0], 0], [0, sin_alpha[0], cos_alpha[0], 0], [0, 0, 0, 1]]))
    M22 = sp.Matrix([[c2, -s2, 0, 0], [s2, c2, 0, 0], [0, 0, 1, d[1]], [0, 0, 0, 1]]) * sp.Matrix(([[1, 0, 0, a[1]], [0, cos_alpha[1], -sin_alpha[1], 0], [0, sin_alpha[1], cos_alpha[1], 0], [0, 0, 0, 1]]))
    M23 = sp.Matrix([[c3, -s3, 0, 0], [s3, c3, 0, 0], [0, 0, 1, d[2]], [0, 0, 0, 1]]) * sp.Matrix(([[1, 0, 0, a[2]], [0, cos_alpha[2], -sin_alpha[2], 0], [0, sin_alpha[2], cos_alpha[2], 0], [0, 0, 0, 1]]))
    M1 = M11 * M12 * M13
    M21inv = M21.inv()
    M22inv = M22.inv()
    M23inv = M23.inv()
    for i in range(4):
            for j in range(4):
                M21inv[i, j] = sp.factor(M21inv[i, j])
                M21inv[i, j] = M21inv[i, j].subs(c1**2+s1**2, 1)
                M21inv[i, j] = M21inv[i, j].subs(c2**2+s2**2, 1)
                M21inv[i, j] = M21inv[i, j].subs(c3**2+s3**2, 1)
                M21inv[i, j] = M21inv[i, j].subs(c4**2+s4**2, 1)
                M21inv[i, j] = M21inv[i, j].subs(c5**2+s5**2, 1)
                M21inv[i, j] = M21inv[i, j].subs(c6**2+s6**2, 1)
    for i in range(4):
            for j in range(4):
                M22inv[i, j] = sp.factor(M22inv[i, j])
                M22inv[i, j] = M22inv[i, j].subs(c1**2+s1**2, 1)
                M22inv[i, j] = M22inv[i, j].subs(c2**2+s2**2, 1)
                M22inv[i, j] = M22inv[i, j].subs(c3**2+s3**2, 1)
                M22inv[i, j] = M22inv[i, j].subs(c4**2+s4**2, 1)
                M22inv[i, j] = M22inv[i, j].subs(c5**2+s5**2, 1)
                M22inv[i, j] = M22inv[i, j].subs(c6**2+s6**2, -1)
    for i in range(4):
            for j in range(4):
                M23inv[i, j] = sp.factor(M23inv[i, j])
                M23inv[i, j] = M23inv[i, j].subs(c1**2+s1**2, 1)
                M23inv[i, j] = M23inv[i, j].subs(c2**2+s2**2, 1)
                M23inv[i, j] = M23inv[i, j].subs(c3**2+s3**2, 1)
                M23inv[i, j] = M23inv[i, j].subs(c4**2+s4**2, 1)
                M23inv[i, j] = M23inv[i, j].subs(c5**2+s5**2, 1)
                M23inv[i, j] = M23inv[i, j].subs(c6**2+s6**2, 1)
    M2 = M23inv * M22inv * M21inv
    # Computing the final matrix [[f1,...,f4], [f5,...,f8], [f9,...,f12], [0, 0, 0, 0]]
    M = M1 - M2 * pose_matrix
    # Taking the equations from M matrix
    for i in range(4):
        for j in range(4):
            list1.append(Poly(M[i, j], c1, c2, c3, c4, c5, c6, s1, s2, s3, s4, s5, s6))
    # Getting rid of the 4 0 equations at the end of the list
    list1 = list1[:12]
    # Computing the c_i^2 + s_i^2 = 1 equations
    list2.append(Poly(c1**2 + s1**2 - 1, c1, s1))
    list2.append(Poly(c2**2 + s2**2 - 1, c2, s2))
    list2.append(Poly(c3**2 + s3**2 - 1, c3, s3))
    list2.append(Poly(c4**2 + s4**2 - 1, c4, s4))
    list2.append(Poly(c5**2 + s5**2 - 1, c5, s5))
    list2.append(Poly(c6**2 + s6**2 - 1, c6, s6))
    # Creating the final list of polynomial equations
    list1 = list1 + list2
    return list1


def get_ikt_gb_lex(eqs):
    file = open("compute_gb.txt", "r")
    lines = file.readlines()
    lines[5] = "eqX := " + str([eq.as_expr() for eq in eqs]) + ";\n"  # put eqs into the 6th line of compute_gb.txt
    file = open("compute_gb.txt", "w")
    file.writelines(lines)
    file.close()
    os.system('cmaple compute_gb.txt')
    with open("gb.txt") as file:
        basis = file.readline()
    return [Poly(eq) for eq in sympify(basis.replace('^', '**'))]


def real_sol(my_list):
    list_new = []
    for i in range(len(my_list)):
        if (my_list[i]).imag == 0.0:
            list_new.append(my_list[i])
    return list_new


def solve_ikt_gb_lex(gb):
    c1, c2, c3, c4, c5, c6, s1, s2, s3, s4, s5, s6 = symbols("c1, c2, c3, c4, c5, c6, s1, s2, s3, s4, s5, s6")
    c6_sol = []
    c5_sol = []
    c4_sol = []
    c3_sol = []
    c2_sol = []
    c1_sol = []
    s5_sol = []
    s4_sol = []
    s3_sol = []
    s2_sol = []
    s1_sol = []
    # Finding solutions for s6
    coeffs = gb[0].all_coeffs()
    s6_sol = np.roots(coeffs)
    s6_sol = list(s6_sol)
    s6_sol = real_sol(s6_sol)
    for i in range(len(s6_sol)):
        c6_sol.append(float(np.roots(((Poly(gb[1].subs(s6, s6_sol[i]))).all_coeffs()))))
        s5_sol.append(float(np.roots(((Poly(gb[2].subs(s6, s6_sol[i]))).all_coeffs()))))
        c5_sol.append(float(np.roots(((Poly(gb[3].subs(s6, s6_sol[i]))).all_coeffs()))))
        s4_sol.append(float(np.roots(((Poly(gb[4].subs(s6, s6_sol[i]))).all_coeffs()))))
        c4_sol.append(float(np.roots(((Poly(gb[5].subs(s6, s6_sol[i]))).all_coeffs()))))
        s3_sol.append(float(np.roots(((Poly(gb[6].subs(s6, s6_sol[i]))).all_coeffs()))))
        c3_sol.append(float(np.roots(((Poly(gb[7].subs(s6, s6_sol[i]))).all_coeffs()))))
        s2_sol.append(float(np.roots(((Poly(gb[8].subs(s6, s6_sol[i]))).all_coeffs()))))
        c2_sol.append(float(np.roots(((Poly(gb[9].subs(s6, s6_sol[i]))).all_coeffs()))))
        s1_sol.append(float(np.roots(((Poly(gb[10].subs(s6, s6_sol[i]))).all_coeffs()))))
        c1_sol.append(float(np.roots(((Poly(gb[11].subs(s6, s6_sol[i]))).all_coeffs()))))
        s6_sol[i] = np.real(s6_sol[i])
    sol_1 = [c1_sol[0], s1_sol[0], c2_sol[0], s2_sol[0], c3_sol[0], s3_sol[0], c4_sol[0], s4_sol[0], c5_sol[0], s5_sol[0], c6_sol[0], s6_sol[0]]
    sol_2 = [c1_sol[1], s1_sol[1], c2_sol[1], s2_sol[1], c3_sol[1], s3_sol[1], c4_sol[1], s4_sol[1], c5_sol[1], s5_sol[1], c6_sol[1], s6_sol[1]]
    sol_3 = [c1_sol[2], s1_sol[2], c2_sol[2], s2_sol[2], c3_sol[2], s3_sol[2], c4_sol[2], s4_sol[2], c5_sol[2], s5_sol[2], c6_sol[2], s6_sol[2]]
    sol_4 = [c1_sol[3], s1_sol[3], c2_sol[3], s2_sol[3], c3_sol[3], s3_sol[3], c4_sol[3], s4_sol[3], c5_sol[3], s5_sol[3], c6_sol[3], s6_sol[3]]
    sol_5 = [c1_sol[4], s1_sol[4], c2_sol[4], s2_sol[4], c3_sol[4], s3_sol[4], c4_sol[4], s4_sol[4], c5_sol[4], s5_sol[4], c6_sol[4], s6_sol[4]]
    sol_6 = [c1_sol[5], s1_sol[5], c2_sol[5], s2_sol[5], c3_sol[5], s3_sol[5], c4_sol[5], s4_sol[5], c5_sol[5], s5_sol[5], c6_sol[5], s6_sol[5]]
    sol_7 = [c1_sol[6], s1_sol[6], c2_sol[6], s2_sol[6], c3_sol[6], s3_sol[6], c4_sol[6], s4_sol[6], c5_sol[6], s5_sol[6], c6_sol[6], s6_sol[6]]
    sol_8 = [c1_sol[7], s1_sol[7], c2_sol[7], s2_sol[7], c3_sol[7], s3_sol[7], c4_sol[7], s4_sol[7], c5_sol[7], s5_sol[7], c6_sol[7], s6_sol[7]]
    my_list = [sol_1, sol_2, sol_3, sol_4, sol_5, sol_6, sol_7, sol_8]
    return my_list

# Creating the json results
"""c1, c2, c3, c4, c5, c6, s1, s2, s3, s4, s5, s6 = symbols("c1, c2, c3, c4, c5, c6, s1, s2, s3, s4, s5, s6")

q = [0.04562910314548819, 0.874218541351503, 0.47744565186582655, 0.0755352660537115]
t = [0.05196408991508393, 0.007931422034575687, 0.8286087511170005]
theta_offset = [0, -math.pi/2, 0, 0, math.pi/2, 0]
d = [0.45, 0, 0, 0.64, 0, 0.2]
a = [0.15, 0.614, 0.2, 0, 0.03, 0]
alpha = [-math.pi/2, 0, -math.pi/2, math.pi/2, -math.pi/2, 0]
my_mechanism = {"theta1 offset": theta_offset[0], "theta2 offset": theta_offset[1], "theta3 offset": theta_offset[2], "theta4 offset": theta_offset[3], "theta5 offset": theta_offset[4], "theta6 offset": theta_offset[5],
                "d1": d[0], "d2": d[1], "d3": d[2], "d4": d[3], "d5": d[4], "d6": d[5],
                "a1": a[0], "a2": a[1], "a3": a[2], "a4": a[3], "a5": a[4], "a6": a[5],
                "alpha1": alpha[0], "alpha2": alpha[1], "alpha3": alpha[2], "alpha4": alpha[3], "alpha5": alpha[4], "alpha6": alpha[5]}
my_pose = {"q": q, "t": t}
my_tol = 10**(-5)

list_of_eqs = ikt_eqs(my_mechanism, my_pose, my_tol)
my_eqs = [s6**16-153.79567958352909730*s6**15+9721.7380230906282752*s6**14-325634.91458497773228*s6**13+6232855.1479750541724*s6**12-69022327.134998457924*s6**11+430354144.68668486332*s6**10-1348388991.7885390647*s6**9+1255428320.8125692091*s6**8+1954226915.0179297153*s6**7-1028030401.5810224124*s6**6-892034939.60910423758*s6**5+254231348.38249667614*s6**4+156435953.36209863672*s6**3-25507771.595339908835*s6**2-8366592.3592402947149*s6+1319932.1484668357200, -.68151170205425542615e-7*s6**15+.10488204104455537951e-4*s6**14-.66360067407104888731e-3*s6**13+.22258907632791584342e-1*s6**12-.42700131713544582615*s6**11+4.7464378648538863128*s6**10-29.797282193272913338*s6**9+94.779954898812582585*s6**8-94.315255768244415942*s6**7-126.51076390168691078*s6**6+86.281431634872046715*s6**5+55.886760010903653525*s6**4-27.238591621694502652*s6**3-9.7819197081188858750*s6**2+c6+4.7194977013817167010*s6+.54437678022865072385, .18670395398100542126e-4*s6**15-.28805324078761138901e-2*s6**14+.18291354729586157806*s6**13-6.1689325657946975605*s6**12+119.37781149132447019*s6**11-1346.8696561249814598*s6**10+8691.2335249812040400*s6**9-29407.671513242170377*s6**8+37742.636331754718371*s6**7+18206.709002731690255*s6**6-28214.466183848224596*s6**5-3022.6798805737638221*s6**4+6303.1672256669887358*s6**3-110.82462865552589224*s6**2+s5-432.44231113431116612*s6+50.976284306912521718, .26107286868039795553e-5*s6**15-.40257183043269038174e-3*s6**14+.25543213455034464139e-1*s6**13-.86044967020825565155*s6**12+16.619506329962271992*s6**11-186.90700112577431428*s6**10+1199.0193182523284017*s6**9-4004.8820013816317179*s6**8+4898.5477879934953169*s6**7+3112.3061646065423633*s6**6-3943.4908792158934773*s6**5-680.09923796131458277*s6**4+940.36220882076325399*s6**3+6.1993639783923229672*s6**2+c5-69.692334439961638713*s6+8.8019400426571060015, -.20461661627973465566e-5*s6**15+.31513886026694677138e-3*s6**14-.19961063391027962446e-1*s6**13+.67064773785143072922*s6**12-12.898841180747356641*s6**11+144.00921853804330700*s6**10-911.24120275261976210*s6**9+2948.8689223557635596*s6**8-3151.3126318354089479*s6**7-3532.9103356400172432*s6**6+3224.3716117594941483*s6**5+1081.6098280848598562*s6**4-901.13943987677824284*s6**3-67.125889324945529616*s6**2+s4+75.586315092551852404*s6-7.5699048115066169726, .16999751958237478809e-3*s6**15-.26227018434619834425e-1*s6**14+1.6653422448002286551*s6**13-56.161630811769284786*s6**12+1086.6976205458323600*s6**11-12258.434897796544698*s6**10+79077.448726724233742*s6**9-267380.95800292155228*s6**8+342306.00413466005575*s6**7+167758.57713212631974*s6**6-256666.14459046551306*s6**5-28951.913449847838600*s6**4+57748.761926006703835*s6**3-870.35292930043378281*s6**2+c4-4020.2634753960159134*s6+480.82435912623524764, .12552182698022927110e-5*s6**15-.19252765300535942128e-3*s6**14+.12122273833642659520e-1*s6**13-.40357907416854431217*s6**12+7.6471996389289183085*s6**11-83.155067172712685054*s6**10+499.54886721739003469*s6**9-1414.2270471414650017*s6**8+513.12457315552407987*s6**7+4398.0996599205627768*s6**6-2249.6521151451407576*s6**5-1612.3363746664292205*s6**4+914.29964688907174295*s6**3+105.67388997501077734*s6**2+s3-97.735937767294269678*s6+10.302072027831626880, .61286301829987545315e-5*s6**15-.94196050727761262089e-3*s6**14+.59487365178104753387e-1*s6**13-1.9896110063462763855*s6**12+37.986337043833275549*s6**11-418.67281688326254068*s6**10+2584.1450006138658996*s6**9-7867.3286683797573416*s6**8+5958.9052484769717788*s6**7+16089.884760911146622*s6**6-10456.824840298362902*s6**5-5477.5885751937497298*s6**4+3664.3345350237636856*s6**3+319.99639662851863414*s6**2+c3-364.61771615904410079*s6+41.297028477017358861, .18820340540595583125e-5*s6**15-.28928417385689764189e-3*s6**14+.18270781772686269242e-1*s6**13-.61117042878312924201*s6**12+11.671381012480046830*s6**11-128.69056638750725281*s6**10+794.94133971182700819*s6**9-2425.0661288153190541*s6**8+1862.2247554680517356*s6**7+4891.9110916956451240*s6**6-3208.5918046901994219*s6**5-1661.9708650983577386*s6**4+1118.2224973346820373*s6**3+97.607987915458597756*s6**2+s2-110.80363479611475145*s6+12.762331169952992936, -.78356294141849142164e-5*s6**15+.12047970670582516588e-2*s6**14-.76129548667005260882e-1*s6**13+2.5484428544970523824*s6**12-48.724926092162250556*s6**11+538.37561393943231821*s6**10-3339.2927540452740839*s6**9+10293.228072836566509*s6**8-8459.9158803249909804*s6**7-19262.556710782556021*s6**6+13246.105859013452369*s6**5+6424.4104469503999734*s6**4-4491.6436105505637125*s6**3-362.38140280594454758*s6**2+c2+438.74568853907972113*s6-49.600940472889810450, .83309932855097002249e-4*s6**15-.12851597644201388293e-1*s6**14+.81591707792513026475*s6**13-27.509575369021923785*s6**12+532.10519245889891836*s6**11-5998.7954968362035385*s6**10+38656.112076154379823*s6**9-130413.19912697011493*s6**8+165682.95751146816470*s6**7+84762.553586362308646*s6**6-124417.49799230434056*s6**5-15609.437856042184573*s6**4+28006.349643031179454*s6**3-212.02062153081412810*s6**2+s1-1951.0984410417583364*s6+227.46487820436224311, .15131771565097411086e-3*s6**15-.23343635266180085042e-1*s6**14+1.4821214081706709789*s6**13-49.975998225658858180*s6**12+966.80471519234870175*s6**11-10902.187685180624236*s6**10+70285.655143156104933*s6**9-237360.17371416095324*s6**8+302675.86860325610933*s6**7+151282.70721126445894*s6**6-225858.31140127773067*s6**5-27039.476802416004033*s6**4+50551.542575500199200*s6**3-529.72649088862698444*s6**2+c1-3500.6654743608472606*s6+410.91420963003288485]

my_eqs[0] = Poly(my_eqs[0], s6)
my_eqs[1] = Poly(my_eqs[1], c6, s6)
my_eqs[2] = Poly(my_eqs[2], s5, s6)
my_eqs[3] = Poly(my_eqs[3], c5, s6)
my_eqs[4] = Poly(my_eqs[4], s4, s6)
my_eqs[5] = Poly(my_eqs[5], c4, s6)
my_eqs[6] = Poly(my_eqs[6], s3, s6)
my_eqs[7] = Poly(my_eqs[7], c3, s6)
my_eqs[8] = Poly(my_eqs[8], s2, s6)
my_eqs[9] = Poly(my_eqs[9], c2, s6)
my_eqs[10] = Poly(my_eqs[10], s1, s6)
my_eqs[11] = Poly(my_eqs[11], c1, s6)


my_solutions = solve_ikt_gb_lex(my_eqs)


sol_1 = my_solutions[0]
sol_2 = my_solutions[1]
sol_3 = my_solutions[2]
sol_4 = my_solutions[3]
sol_5 = my_solutions[4]
sol_6 = my_solutions[5]
sol_7 = my_solutions[6]
sol_8 = my_solutions[7]


sol1 = {'theta1':  math.atan2(sol_1[1], sol_1[0]) - my_mechanism['theta1 offset'], 'theta2':  math.atan2(sol_1[3], sol_1[2]) - my_mechanism['theta2 offset'],
        'theta3':  math.atan2(sol_1[5], sol_1[4]) - my_mechanism['theta3 offset'], 'theta4':  math.atan2(sol_1[7], sol_1[6]) - my_mechanism['theta4 offset'],
        'theta5':  math.atan2(sol_1[9], sol_1[8]) - my_mechanism['theta5 offset'], 'theta6':  math.atan2(sol_1[11], sol_1[10]) - my_mechanism['theta6 offset']}
sol2 = {'theta1':  math.atan2(sol_2[1], sol_2[0]) - my_mechanism['theta1 offset'], 'theta2':  math.atan2(sol_2[3], sol_2[2]) - my_mechanism['theta2 offset'],
        'theta3':  math.atan2(sol_2[5], sol_2[4]) - my_mechanism['theta3 offset'], 'theta4':  math.atan2(sol_2[7], sol_2[6]) - my_mechanism['theta4 offset'],
        'theta5':  math.atan2(sol_2[9], sol_2[8]) - my_mechanism['theta5 offset'], 'theta6':  math.atan2(sol_2[11], sol_2[10]) - my_mechanism['theta6 offset']}
sol3 = {'theta1':  math.atan2(sol_3[1], sol_3[0]) - my_mechanism['theta1 offset'], 'theta2':  math.atan2(sol_3[3], sol_3[2]) - my_mechanism['theta2 offset'],
        'theta3':  math.atan2(sol_3[5], sol_3[4]) - my_mechanism['theta3 offset'], 'theta4':  math.atan2(sol_3[7], sol_3[6]) - my_mechanism['theta4 offset'],
        'theta5':  math.atan2(sol_3[9], sol_3[8]) - my_mechanism['theta5 offset'], 'theta6':  math.atan2(sol_3[11], sol_3[10]) - my_mechanism['theta6 offset']}
sol4 = {'theta1':  math.atan2(sol_4[1], sol_4[0]) - my_mechanism['theta1 offset'], 'theta2':  math.atan2(sol_4[3], sol_4[2]) - my_mechanism['theta2 offset'],
        'theta3':  math.atan2(sol_4[5], sol_4[4]) - my_mechanism['theta3 offset'], 'theta4':  math.atan2(sol_4[7], sol_4[6]) - my_mechanism['theta4 offset'],
        'theta5':  math.atan2(sol_4[9], sol_4[8]) - my_mechanism['theta5 offset'], 'theta6':  math.atan2(sol_4[11], sol_4[10]) - my_mechanism['theta6 offset']}
sol5 = {'theta1':  math.atan2(sol_5[1], sol_5[0]) - my_mechanism['theta1 offset'], 'theta2':  math.atan2(sol_5[3], sol_5[2]) - my_mechanism['theta2 offset'],
        'theta3':  math.atan2(sol_5[5], sol_5[4]) - my_mechanism['theta3 offset'], 'theta4':  math.atan2(sol_5[7], sol_5[6]) - my_mechanism['theta4 offset'],
        'theta5':  math.atan2(sol_5[9], sol_5[8]) - my_mechanism['theta5 offset'], 'theta6':  math.atan2(sol_5[11], sol_5[10]) - my_mechanism['theta6 offset']}
sol6 = {'theta1':  math.atan2(sol_6[1], sol_6[0]) - my_mechanism['theta1 offset'], 'theta2':  math.atan2(sol_6[3], sol_6[2]) - my_mechanism['theta2 offset'],
        'theta3':  math.atan2(sol_6[5], sol_6[4]) - my_mechanism['theta3 offset'], 'theta4':  math.atan2(sol_6[7], sol_6[6]) - my_mechanism['theta4 offset'],
        'theta5':  math.atan2(sol_6[9], sol_6[8]) - my_mechanism['theta5 offset'], 'theta6':  math.atan2(sol_6[11], sol_6[10]) - my_mechanism['theta6 offset']}
sol7 = {'theta1':  math.atan2(sol_7[1], sol_7[0]) - my_mechanism['theta1 offset'], 'theta2':  math.atan2(sol_7[3], sol_7[2]) - my_mechanism['theta2 offset'],
        'theta3':  math.atan2(sol_7[5], sol_7[4]) - my_mechanism['theta3 offset'], 'theta4':  math.atan2(sol_7[7], sol_7[6]) - my_mechanism['theta4 offset'],
        'theta5':  math.atan2(sol_7[9], sol_7[8]) - my_mechanism['theta5 offset'], 'theta6':  math.atan2(sol_7[11], sol_7[10]) - my_mechanism['theta6 offset']}
sol8 = {'theta1':  math.atan2(sol_8[1], sol_8[0]) - my_mechanism['theta1 offset'], 'theta2':  math.atan2(sol_8[3], sol_8[2]) - my_mechanism['theta2 offset'],
        'theta3':  math.atan2(sol_8[5], sol_8[4]) - my_mechanism['theta3 offset'], 'theta4':  math.atan2(sol_8[7], sol_8[6]) - my_mechanism['theta4 offset'],
        'theta5':  math.atan2(sol_8[9], sol_8[8]) - my_mechanism['theta5 offset'], 'theta6':  math.atan2(sol_8[11], sol_8[10]) - my_mechanism['theta6 offset']}

sol11 = {'theta1':  math.atan2(np.sin(sol1['theta1']), np.cos(sol1['theta1'])), 'theta2':  math.atan2(np.sin(sol1['theta2']), np.cos(sol1['theta2'])),
         'theta3':  math.atan2(np.sin(sol1['theta3']), np.cos(sol1['theta3'])), 'theta4':  math.atan2(np.sin(sol1['theta4']), np.cos(sol1['theta4'])),
         'theta5':  math.atan2(np.sin(sol1['theta5']), np.cos(sol1['theta5'])), 'theta6':  math.atan2(np.sin(sol1['theta6']), np.cos(sol1['theta6']))}
sol22 = {'theta1':  math.atan2(np.sin(sol2['theta1']), np.cos(sol2['theta1'])), 'theta2':  math.atan2(np.sin(sol2['theta2']), np.cos(sol2['theta2'])),
         'theta3':  math.atan2(np.sin(sol2['theta3']), np.cos(sol2['theta3'])), 'theta4':  math.atan2(np.sin(sol2['theta4']), np.cos(sol2['theta4'])),
         'theta5':  math.atan2(np.sin(sol2['theta5']), np.cos(sol2['theta5'])), 'theta6':  math.atan2(np.sin(sol2['theta6']), np.cos(sol2['theta6']))}
sol33 = {'theta1':  math.atan2(np.sin(sol3['theta1']), np.cos(sol3['theta1'])), 'theta2':  math.atan2(np.sin(sol3['theta2']), np.cos(sol3['theta2'])),
         'theta3':  math.atan2(np.sin(sol3['theta3']), np.cos(sol3['theta3'])), 'theta4':  math.atan2(np.sin(sol3['theta4']), np.cos(sol3['theta4'])),
         'theta5':  math.atan2(np.sin(sol3['theta5']), np.cos(sol3['theta5'])), 'theta6':  math.atan2(np.sin(sol3['theta6']), np.cos(sol3['theta6']))}
sol44 = {'theta1':  math.atan2(np.sin(sol4['theta1']), np.cos(sol4['theta1'])), 'theta2':  math.atan2(np.sin(sol4['theta2']), np.cos(sol4['theta2'])),
         'theta3':  math.atan2(np.sin(sol4['theta3']), np.cos(sol4['theta3'])), 'theta4':  math.atan2(np.sin(sol4['theta4']), np.cos(sol4['theta4'])),
         'theta5':  math.atan2(np.sin(sol4['theta5']), np.cos(sol4['theta5'])), 'theta6':  math.atan2(np.sin(sol4['theta6']), np.cos(sol4['theta6']))}
sol55 = {'theta1':  math.atan2(np.sin(sol5['theta1']), np.cos(sol5['theta1'])), 'theta2':  math.atan2(np.sin(sol5['theta2']), np.cos(sol5['theta2'])),
         'theta3':  math.atan2(np.sin(sol5['theta3']), np.cos(sol5['theta3'])), 'theta4':  math.atan2(np.sin(sol5['theta4']), np.cos(sol5['theta4'])),
         'theta5':  math.atan2(np.sin(sol5['theta5']), np.cos(sol5['theta5'])), 'theta6':  math.atan2(np.sin(sol5['theta6']), np.cos(sol5['theta6']))}
sol66 = {'theta1':  math.atan2(np.sin(sol6['theta1']), np.cos(sol6['theta1'])), 'theta2':  math.atan2(np.sin(sol6['theta2']), np.cos(sol6['theta2'])),
         'theta3':  math.atan2(np.sin(sol6['theta3']), np.cos(sol6['theta3'])), 'theta4':  math.atan2(np.sin(sol6['theta4']), np.cos(sol6['theta4'])),
         'theta5':  math.atan2(np.sin(sol6['theta5']), np.cos(sol6['theta5'])), 'theta6':  math.atan2(np.sin(sol6['theta6']), np.cos(sol6['theta6']))}
sol77 = {'theta1':  math.atan2(np.sin(sol7['theta1']), np.cos(sol7['theta1'])), 'theta2':  math.atan2(np.sin(sol7['theta2']), np.cos(sol7['theta2'])),
         'theta3':  math.atan2(np.sin(sol7['theta3']), np.cos(sol7['theta3'])), 'theta4':  math.atan2(np.sin(sol7['theta4']), np.cos(sol7['theta4'])),
         'theta5':  math.atan2(np.sin(sol7['theta5']), np.cos(sol7['theta5'])), 'theta6':  math.atan2(np.sin(sol7['theta6']), np.cos(sol7['theta6']))}
sol88 = {'theta1':  math.atan2(np.sin(sol8['theta1']), np.cos(sol8['theta1'])), 'theta2':  math.atan2(np.sin(sol8['theta2']), np.cos(sol8['theta2'])),
         'theta3':  math.atan2(np.sin(sol8['theta3']), np.cos(sol8['theta3'])), 'theta4':  math.atan2(np.sin(sol8['theta4']), np.cos(sol8['theta4'])),
         'theta5':  math.atan2(np.sin(sol8['theta5']), np.cos(sol8['theta5'])), 'theta6':  math.atan2(np.sin(sol8['theta6']), np.cos(sol8['theta6']))}



real_sols = [sol11, sol22, sol33, sol44, sol55, sol66, sol77, sol88]
import json
with open("hw08.json", "w") as outfile:
    json.dump(real_sols, outfile)"""