from numpy import array, sin, cos


def inv_pend_ode_simple(t, x, u):
  # params, hardcoded
  m = 0.1
  L = 0.5
  J = 0.008
  g = 9.81
   
  # unpack x
  theta, theta_dot = x
  Delta = (J + m*L**2) 

  # dynamics
  f = array([theta_dot,
             1/Delta * ( m * g * L * sin(theta) )]) 

  G = array([0,
             1/Delta * ( - m * L * cos(theta) )])
  
  return f + G*u
