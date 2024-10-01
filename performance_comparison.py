import numpy as np
import os
from dill import load, dump
from operator import itemgetter

from ruku4 import ruku4_step
from plants import inv_pend_ode_simple
from controllers import SontagController, LQRController, FeedbackLinController

from cost_fcn import cost_sum_of_traj

from tqdm import tqdm


if __name__ == "__main__":
  export_folder_etc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "generated")) # export folder for clfs, etc
  export_folder_sims = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "sim_results")) # export folder for sim results


  with open(os.path.join(f"{export_folder_etc}", "b_expr"), 'rb') as file:
    b_expr = load(file)

  with open(os.path.join(f"{export_folder_etc}", "controller_sontag_expr"), 'rb') as file:
    cntr_sontag_expr = load(file)

  with open(os.path.join(f"{export_folder_etc}", "controller_lqr_expr"), 'rb') as file:
    cntr_lqr_expr = load(file)

  with open(os.path.join(f"{export_folder_etc}", "controller_feedb_lin_expr"), 'rb') as file:
    cntr_feedb_lin_expr = load(file)

  cntr_sontag = SontagController(b_expr=b_expr, controller_expr=cntr_sontag_expr)
  cntr_lqr = LQRController(controller_expr=cntr_lqr_expr)
  cntr_feedb_lin = FeedbackLinController(controller_expr=cntr_feedb_lin_expr)

  n_x, n_u = cntr_sontag.get_dims()

  h = 0.01
  t_span = np.arange(0, 15, h)

  with open(os.path.join(f"{export_folder_etc}", "riccati_design_matrices"), 'rb') as file:
    riccati_matrices = load(file) # first load into temporary (error otherwise)
    Q, R, P, K = itemgetter('Q', 'R', 'P', 'K')(riccati_matrices)
    del riccati_matrices
 
  # number of initial conditions in theta and x
  num_theta_0_pts = int(1e3)
  # arrays of initial conditions
  theta_0_max = 89* 2*np.pi/360
  theta_0_array = np.linspace(0, theta_0_max, num_theta_0_pts)

  # Create a grid of theta values
  theta_0_grid = np.arange(start=-theta_0_max, stop=theta_0_max, step=2*theta_0_max/num_theta_0_pts)

  # 3d matrix of initial conditions
  # x_0_array = np.array([theta_0_grid, np.zeros((3, num_theta_0_pts))])
  x_0_array = np.concatenate((theta_0_grid[:, None].T, np.zeros((1, num_theta_0_pts))), axis=0)

  n_sims = num_theta_0_pts

  # preallocate
  X_sim_sontag = np.nan * np.ones((n_x , t_span.size))
  U_sim_sontag = np.nan * np.ones((n_u , t_span.size))

  X_sim_lqr = np.nan * np.ones((n_x , t_span.size))
  U_sim_lqr = np.nan * np.ones((n_u , t_span.size))

  X_sim_feedb_lin = np.nan * np.ones((n_x , t_span.size))
  U_sim_feedb_lin = np.nan * np.ones((n_u , t_span.size))

  cost_sim_sontag = np.nan * np.ones((1, num_theta_0_pts))
  cost_sim_lqr = np.nan * np.ones((1, num_theta_0_pts))
  cost_sim_feedb_lin = np.nan * np.ones((1, num_theta_0_pts))

  # iterate over all 2d grid points of initial conditions
  pbar = tqdm(total=n_sims)
  for i_theta in range(num_theta_0_pts):
    pbar.update(1)
    x_0 = x_0_array[:, i_theta]

    X_sim_sontag[:, 0] = x_0
    X_sim_lqr[:, 0] = x_0
    X_sim_feedb_lin[:, 0] = x_0

    # Sim: LQR
    for index, time in enumerate(t_span):
      x = X_sim_lqr[:, index]
      u = cntr_lqr.eval(x)[0][0]

      x_next = ruku4_step(f=inv_pend_ode_simple, t=time, x=x, h=h, u=u)

      U_sim_lqr[:, index] = u
      if time < t_span[-1]:
        X_sim_lqr[:, index + 1] = x_next

    # Sim: Feedback Linearization
    for index, time in enumerate(t_span):
      x = X_sim_feedb_lin[:, index]
      u = cntr_feedb_lin.eval(x)[0][0]

      x_next = ruku4_step(f=inv_pend_ode_simple, t=time, x=x, h=h, u=u)

      U_sim_feedb_lin[:, index] = u
      if time < t_span[-1]:
        X_sim_feedb_lin[:, index + 1] = x_next

    # Sim: Sontag
    for index, time in enumerate(t_span):
      x = X_sim_sontag[:, index]
      u = cntr_sontag.eval(x)[0][0]

      x_next = ruku4_step(f=inv_pend_ode_simple, t=time, x=x, h=h, u=u)

      U_sim_sontag[:, index] = u
      if time < t_span[-1]:
        X_sim_sontag[:, index + 1] = x_next
    
    # evaluate
    cost_sim_sontag[:, i_theta] = cost_sum_of_traj(Q, R, h, x_traj=X_sim_sontag, u_traj=U_sim_sontag)
    cost_sim_lqr[:, i_theta] = cost_sum_of_traj(Q, R, h, x_traj=X_sim_lqr, u_traj=U_sim_lqr)
    cost_sim_feedb_lin[:, i_theta] = cost_sum_of_traj(Q, R, h, x_traj=X_sim_feedb_lin, u_traj=U_sim_feedb_lin)


# save riccati design choices Q, R, P, K
with open(os.path.join(export_folder_sims, "performance_comparison_sim_theta"), "wb") as file:
    dump({'cost_sim_sontag': cost_sim_sontag, 'cost_sim_lqr': cost_sim_lqr, 'cost_sim_feedb_lin': cost_sim_feedb_lin, 'theta_0_grid': theta_0_grid}, file)
