import numpy as np
import matplotlib.pyplot as plt
import os
from dill import load
from operator import itemgetter

plt.rcParams['text.usetex'] = True

## load sim data
export_folder_sims = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "sim_results")) # export folder for sim results

with open(os.path.join(f"{export_folder_sims}", "performance_comparison_sim_theta"), 'rb') as file:
  sim_result_matrices = load(file) # first load into temporary (error otherwise)
  cost_sim_sontag, cost_sim_lqr, cost_sim_feedb_lin, theta_0_grid = itemgetter('cost_sim_sontag', 'cost_sim_lqr', 'cost_sim_feedb_lin', 'theta_0_grid')(sim_result_matrices)
  del sim_result_matrices

# cost_ratio = np.divide(cost_sim_sontag, cost_sim_lqr)
cost_ratio_lqr = np.minimum(np.divide(cost_sim_sontag, cost_sim_lqr), np.ones_like(cost_sim_sontag)*10)
cost_ratio_feedb_lin = np.minimum(np.divide(cost_sim_sontag, cost_sim_feedb_lin), np.ones_like(cost_sim_sontag)*10)

cost_ratio_lqr_above_one = cost_ratio_lqr.copy()
cost_ratio_feedb_lin_above_one = cost_ratio_feedb_lin.copy()

above_threshold_indices_lqr = np.where(cost_ratio_lqr > 1)
above_threshold_indices_feedb_lin = np.where(cost_ratio_feedb_lin > 1)

cost_ratio_lqr_above_one[cost_ratio_lqr_above_one <= 1] = np.nan
cost_ratio_feedb_lin_above_one[cost_ratio_feedb_lin_above_one <= 1] = np.nan

## Plotting

fig1, ax1 = plt.subplots()

ax1.plot(theta_0_grid, cost_sim_sontag[0, :], label='Sontag', color='blue')
ax1.plot(theta_0_grid, cost_sim_lqr[0, :], label='LQR', color='red')
ax1.plot(theta_0_grid, cost_sim_feedb_lin[0, :], label='FBL', color='green')

# Set labels and title
ax1.set_xlabel(r'$\theta_0$')
ax1.set_ylabel(r'cost over $\theta_0$')
# Add a legend
ax1.legend()


fig2, ax2 = plt.subplots()

# ax2.plot(theta_0_grid, cost_ratio[0, :], color='orange')

ax2.plot(theta_0_grid, cost_ratio_lqr[0, :], color='blue', label='to LQR')
ax2.plot(theta_0_grid, cost_ratio_feedb_lin[0, :], color='green', label='to FBL')
#ax2.plot(theta_0_grid, cost_ratio_lqr_above_one[0, :], color='red')


ax2.axhline(y=1, color='k', linestyle='--')

# Set labels and title
ax2.set_xlabel(r'$\theta_0$')
ax2.set_ylabel(r'Cost ratio')
# ax2.set_ylabel(r'$J_{\mathrm{S}}(\theta_0)$/$J_{\mathrm{X}}(\theta_0)$')
ax2.grid()
# Add a legend
ax2.legend()


fig3, ax3 = plt.subplots()

ax3.plot(theta_0_grid, cost_ratio_lqr[0, :], color='blue', label='to LQR')
ax3.plot(theta_0_grid, cost_ratio_feedb_lin[0, :], color='green', label='to FBL')
#ax3.plot(theta_0_grid, cost_ratio_lqr_above_one[0, :], color='red')


ax3.axhline(y=1, color='k', linestyle='--')
ax3.set_xlim(left=-0.9, right=0.9)
ax3.set_ylim(bottom=0.97, top=1.002)

# Set labels and title
ax3.set_xlabel(r'$\theta_0$')
ax2.set_ylabel(r'Cost ratio (Zoomed in)')
# ax3.set_ylabel(r'$J_{\mathrm{S}}(\theta_0)$/$J_{\mathrm{X}}(\theta_0)$')
ax3.grid()
# Add a legend
ax3.legend()


# Show the plot
plt.show()
