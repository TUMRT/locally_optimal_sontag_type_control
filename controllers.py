# This only implements calls to generated controller functions

from numpy import ones


class SontagController:

  def __init__(self, b_expr, controller_expr):
    self.b = b_expr
    self.controller = controller_expr

    self.n_x = len(self.controller.__code__.co_varnames) - 1

    # Determine the output dimensions
    sample_input = 0.1 * ones((self.n_x, 1))  # Provide a sample input
    self.n_u = len(self.controller(sample_input))

  def get_dims(self):
    return self.n_x, self.n_u

  def eval(self, x): # -> float
    if all(self.b(x) == 0):
      u = [[0] * self.n_u]
    else:
      u = self.controller(x)
    return u


class LQRController:

  def __init__(self, controller_expr):
    self.controller = controller_expr

  def eval(self, x): # -> float
    u = self.controller(x)
    return u
  
class FeedbackLinController:

  def __init__(self, controller_expr):
    self.controller = controller_expr

  def eval(self, x): # -> float
    u = self.controller(x)
    return u
