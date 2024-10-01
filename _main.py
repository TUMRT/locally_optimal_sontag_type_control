import numpy as np
import matplotlib.pyplot as plt
import os
from dill import load

from ruku4 import ruku4_step
from plants import inv_pend_ode_simple
from controllers import SontagController, LQRController, FeedbackLinController

import time
from tqdm import tqdm


if __name__ == "__main__":
  plt.rcParams['text.usetex'] = True

  export_folder = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "generated")) # export folder for clfs, etc

  with open(os.path.join(f"{export_folder}", "b_expr"), 'rb') as file:
    b_expr = load(file)

  with open(os.path.join(f"{export_folder}", "controller_sontag_expr"), 'rb') as file:
    cntr_sontag_expr = load(file)

  with open(os.path.join(f"{export_folder}", "controller_lqr_expr"), 'rb') as file:
    cntr_lqr_expr = load(file)

  with open(os.path.join(f"{export_folder}", "controller_feedb_lin_expr"), 'rb') as file:
    cntr_feedb_lin_expr = load(file)

  cntr_sontag = SontagController(b_expr=b_expr, controller_expr=cntr_sontag_expr)
  cntr_lqr = LQRController(controller_expr=cntr_lqr_expr)
  cntr_feedb_lin = FeedbackLinController(controller_expr=cntr_feedb_lin_expr)

  n_x, n_u = cntr_sontag.get_dims()

  h = 0.01
  t_span = np.arange(0, 2, h)

  x_names = [r'$\theta$', r'$\dot{\theta}$']
  u_names = [r'$\ddot{y}$']

  theta_0_deg = 25

  x_0 = np.array([theta_0_deg*2*np.pi/360, 0])

  # preallocate
  X_sim_sontag = np.nan * np.ones((n_x , t_span.size))
  U_sim_sontag = np.nan * np.ones((n_u , t_span.size))

  X_sim_lqr = np.nan * np.ones((n_x , t_span.size))
  U_sim_lqr = np.nan * np.ones((n_u , t_span.size))

  X_sim_feedb_lin = np.nan * np.ones((n_x , t_span.size))
  U_sim_feedb_lin = np.nan * np.ones((n_u , t_span.size))

  X_sim_sontag[:, 0] = x_0
  X_sim_lqr[:, 0] = x_0
  X_sim_feedb_lin[:, 0] = x_0

  t_cntr_sontag = np.nan * np.ones((t_span.size, 1))
  t_cntr_lqr = np.nan * np.ones((t_span.size, 1))
  t_cntr_feedb_lin = np.nan * np.ones((t_span.size, 1))

  # Sim: LQR
  pbar = tqdm(total=t_span.size)
  for index, t in enumerate(t_span):
    x = X_sim_lqr[:, index]

    start = time.time()
    u = cntr_lqr.eval(x)[0][0]
    end = time.time()
    t_cntr_lqr[index] = end - start

    x_next = ruku4_step(f=inv_pend_ode_simple, t=t, x=x, h=h, u=u)

    U_sim_lqr[:, index] = u
    if t < t_span[-1]:
      X_sim_lqr[:, index + 1] = x_next
    
    pbar.update(1)

  # Sim: Feedback Linearization
  pbar = tqdm(total=t_span.size)
  for index, t in enumerate(t_span):
    x = X_sim_feedb_lin[:, index]

    start = time.time()
    u = cntr_feedb_lin.eval(x)[0][0]
    end = time.time()
    t_cntr_feedb_lin[index] = end - start

    x_next = ruku4_step(f=inv_pend_ode_simple, t=t, x=x, h=h, u=u)

    U_sim_feedb_lin[:, index] = u
    if t < t_span[-1]:
      X_sim_feedb_lin[:, index + 1] = x_next
    
    pbar.update(1)

  # Sim: Sontag
  pbar = tqdm(total=t_span.size)
  for index, t in enumerate(t_span):
    x = X_sim_sontag[:, index]

    start = time.time()
    u = cntr_sontag.eval(x)[0][0]
    end = time.time()
    t_cntr_sontag[index] = end - start

    x_next = ruku4_step(f=inv_pend_ode_simple, t=t, x=x, h=h, u=u)

    U_sim_sontag[:, index] = u
    if t < t_span[-1]:
      X_sim_sontag[:, index + 1] = x_next

    pbar.update(1)

  ## Plotting

  colors_x = ['red', 'green']
  colors_u = ['blue']

  # LQR
  for i in range(n_x):
    plt.plot(t_span, X_sim_lqr[i], label=x_names[i] + ' LQR', color=colors_x[i], linestyle='dashed')

  for i in range(n_u):
    plt.plot(t_span, U_sim_lqr[0, :], label=u_names[i] + ' LQR', color=colors_u[i], linestyle='dashed')

  # Feedback linearization
  for i in range(n_x):
    plt.plot(t_span, X_sim_feedb_lin[i], label=x_names[i] + ' FBL', color=colors_x[i], linestyle='dotted')

  for i in range(n_u):
    plt.plot(t_span, U_sim_feedb_lin[0, :], label=u_names[i] + ' FBL', color=colors_u[i], linestyle='dotted')  

  # Sontag
  for i in range(n_x):
    plt.plot(t_span, X_sim_sontag[i], label=x_names[i] + ' Sontag', color=colors_x[i])

  for i in range(n_u):
    plt.plot(t_span, U_sim_sontag[0, :], label=u_names[i] + ' Sontag', color=colors_u[i])

  plt.xlabel(r'$t$ in $s$')

  plt.grid(True, linestyle=':', linewidth=0.5)
  plt.legend()

  plt.show()
