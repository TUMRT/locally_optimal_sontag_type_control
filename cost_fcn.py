# sum running cost up over whole given x and u trajectories
def cost_sum_of_traj(Q, R, h, x_traj, u_traj):
  cost = 0
  for i in range(x_traj.shape[1]):
    x = x_traj[:, i]
    u = u_traj[:, i]
    cost = cost + 0.5 * h * (x.T@Q@x + u.T@R@u)
  
  return cost