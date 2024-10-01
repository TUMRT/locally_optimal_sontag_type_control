## Calculate CLF control laws
# Joscha Bongard, joscha.bongard@tum.de

from sympy import symbols, Matrix, lambdify, sin, cos, sqrt
import numpy as np
from scipy.linalg import solve_continuous_are # for LQR
import dill # for saving lambdified sympy functions
from pathlib import Path

# for exporting lambdified sympy functions
dill.settings['recurse'] = True

# export_folder_controller = str(Path(__file__).parents[0].joinpath('sontag')) # export folder for controllers
export_folder_etc = str(Path(__file__).parent.joinpath('generated')) # export folder for clfs, etc

## sympy symbols

# physical states
theta, theta_dot = symbols('theta theta_dot', real=True)
# physical inputs
y_dd_in = symbols('y_dd_in', real=True)

# write CLFs and their derivatives to matlab functions?
writeCLFs2Fcns = True

## model

print('building model')

m = 0.1
k = 0.1
L = 0.5
J = 0.008
g = 9.81

Delta = (J + m*L**2) 

# dynamics
# Khalil - Nonlinear Control, Appendix A1, p.359
f = Matrix([theta_dot,
            1/Delta * ( m * g * L * sin(theta) )]) 

G = Matrix([0,
            1/Delta * ( - m * L * cos(theta) )])

## coordinates

# inputs
U = Matrix([y_dd_in])

# original physical states (omit path variable s, include physical inputs)
X = Matrix([theta, theta_dot])

# number of states, inputs
n_x = X.shape[0]
n_u = U.shape[0]

# equilibrium
X_eq = np.zeros((n_x, 1))
U_eq = np.zeros((n_u, 1))

# overall dynamics in physical coordinates
F = f + G*U

## Linearization

print('linearizing model')

A_sym = F.jacobian(X)
B_sym = F.jacobian(U)

# compute numeric A, B
A = np.array(A_sym.subs(list(zip(Matrix([X,U]),Matrix([Matrix(X_eq),Matrix(U_eq)])))), dtype=float)
B = np.array(B_sym.subs(list(zip(Matrix([X,U]),Matrix([Matrix(X_eq),Matrix(U_eq)])))), dtype=float)

# optional stuff
eigs = np.linalg.eigvals(A)
nonzero_eigs = eigs[eigs != 0]
stiffness = max(abs(nonzero_eigs))/(min(abs(nonzero_eigs)))
# print(f"Numerical Stiffness: {stiffness}")

## LQR controller

print('Computing LQR')

# LQR weighting matrices
Q = np.eye(n_x)
R = np.eye(n_u)

# compute LQR
P = solve_continuous_are(A, B, Q, R)
K_lqr = np.linalg.inv(R)@B.T@P

if any(eig_cl >= 0 for eig_cl in np.linalg.eigvals(A-B@K_lqr)):
    print("warning: unstable under LQR")

control_law_lqr = -K_lqr * X # LQR control law

# save linear system A, B
with open(str(Path(export_folder_etc).joinpath('lin_system')), "wb") as file:
    dill.dump({'A': A, 'B': B}, file)

# save riccati design choices Q, R, P, K
with open(str(Path(export_folder_etc).joinpath('riccati_design_matrices')), "wb") as file:
    dill.dump({'Q': Q, 'R': R, 'P': P, 'K': K_lqr}, file)


## feedback linearization design

print('computing feedback linearization-based controller')

rel_degr = 2 # hardcoded
h = X[0] # theta is flat output
z = X # transformed state in flatness coordinates

if rel_degr == 0:
    L_f_i_h = [h]
    L_g_L_f_delta_minus_one_h = G
elif rel_degr == 1:
    L_f_i_h = [h, Matrix([h]).jacobian(X) * f]
    L_g_L_f_delta_minus_one_h = h.jacobian(X) * G
else:
    L_f_i_h = [h, Matrix([h]).jacobian(X) * f]
    for i in range(rel_degr - 1):
        L_f_i_h.append(L_f_i_h[-1].jacobian(X) * f)

    L_g_L_f_delta_minus_one_h = L_f_i_h[-2].jacobian(X) * G

K_feedb_lin = - m * g * L / (J + m * L**2) * np.array([1, 0]) - m * L / (J + m * L**2) * K_lqr

artificial_input_v = - K_feedb_lin * z

control_law_feedback_lin = L_g_L_f_delta_minus_one_h.inv() * (-1 * L_f_i_h[-1] + artificial_input_v)

## CLF controller design

print('computing CLF-based controllers')

# Ansatz: Use LQR Lyap. fcn as CLF candidate which works at least locally
# for smooth systems
# plus level sets align locally with Value fcn
V = 1/2*X.T*P*X

# in the following, we convert 1x1 matrices to scalars by indexing

b = V.jacobian(X)*G # = L_G V
a = (V.jacobian(X)*f)[0,0] # = L_f V

# OG Sontag's formula
control_law_sontag = - ((a+sqrt(a**2+(b*b.T)[0,0]*(b*b.T)[0,0]))/(b*b.T)[0,0])*b.T
# Variant by Sackmann: https://www.degruyter.com/document/doi/10.1524/auto.2005.53.8.367/html
lamda = ((a + sqrt(a**2 + (X.T*Q*X)[0,0]*(b*np.linalg.inv(R)*b.T)[0,0] )) / (b*np.linalg.inv(R)*b.T)[0,0])
control_law_inv_optimal = - np.linalg.inv(R)*b.T * lamda

# Choice of CLF controller
control_law_clf = control_law_inv_optimal

## print to file

# exporting lambdified function
# https://stackoverflow.com/questions/29079923/save-load-sympy-lambdifed-expressions

print('exporting expressions')

# export b
with open(str(Path(export_folder_etc).joinpath('b_expr')), "wb") as file:
    dill.dump(lambdify([X], b, 'numpy'), file)

# export sontag controller to target folder
with open(str(Path(export_folder_etc).joinpath('controller_sontag_expr')), "wb") as file:
    dill.dump(lambdify([X], control_law_clf, 'numpy'), file)

# export lqr
with open(str(Path(export_folder_etc).joinpath('controller_lqr_expr')), "wb") as file:
    dill.dump(lambdify([X], control_law_lqr, 'numpy'), file)

# export feedback linearization controller
with open(str(Path(export_folder_etc).joinpath('controller_feedb_lin_expr')), "wb") as file:
    dill.dump(lambdify([X], control_law_feedback_lin, 'numpy'), file)

print('finished computing and writing control laws')

# write CLFs and their time derivatives to matlab functions?
if writeCLFs2Fcns:
    print('writing CLFs to functions')
    # CLF itself
    with open(str(Path(export_folder_etc).joinpath('V')), "wb") as file:
        dill.dump(lambdify([X], V, 'numpy'), file)

    V_dot = (V.jacobian(X)*F)[0,0]
    # derivative of CLF under clf controller
    V_dot_clf = V_dot.subs(list(zip(Matrix([U]),Matrix([Matrix(control_law_clf)]))))
    with open(str(Path(export_folder_etc).joinpath('V_dot_clf')), "wb") as file:
        dill.dump(lambdify([X], V_dot_clf, 'numpy'), file)

    # and under LQR
    V_dot_lqr = V_dot.subs(list(zip(Matrix([U]),Matrix([Matrix(control_law_lqr)]))))
    with open(str(Path(export_folder_etc).joinpath('V_dot_lqr')), "wb") as file:
        dill.dump(lambdify([X], V_dot_lqr, 'numpy'), file)

print('done')
