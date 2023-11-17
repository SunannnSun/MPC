import numpy as np
from math import sin, cos
from scipy.signal import cont2discrete


from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.mathematicalprogram import MathematicalProgram
import pydrake.symbolic as sym


class Quadrotor(object):
  def __init__(self, x_d, Q, R, Qf):
    self.g = 9.81
    self.m = 1
    self.a = 0.25
    self.I = 0.0625
    self.Q = Q
    self.R = R
    self.Qf = Qf

    # Input limits
    self.umin = 0
    self.umax = 5.5

    self.n_x = 6
    self.n_u = 2

    self.x_d = x_d
    self.u_d = np.array([self.m*self.g/2, self.m*self.g/2])
    self.dt  = 0.1
   
  # def x_d(self):
  #   # Nominal state
  #   return np.array([0, 0, 0, 0, 0, 0])

  # def u_d(self):
  #   # Nominal input
  #   return np.array([self.m*self.g/2, self.m*self.g/2])

  def continuous_time_full_dynamics(self, x, u):
    # Dynamics for the quadrotor
    g = self.g
    m = self.m
    a = self.a
    I = self.I

    theta = x[2]
    ydot = x[3]
    zdot = x[4]
    thetadot = x[5]
    u0 = u[0]
    u1 = u[1]

    xdot = np.array([ydot,
                     zdot,
                     thetadot,
                     -sin(theta) * (u0 + u1) / m,
                     -g + cos(theta) * (u0 + u1) / m,
                     a * (u0 - u1) / I])
    return xdot

  def continuous_time_linearized_dynamics(self, x_current, u_current):
    # Dynamics linearized at the fixed point


    # Linearize about the origin
    """
    A = np.zeros((6,6))
    A[:3, -3:] = np.identity(3)
    A[3, 2] = -self.g
    
    B = np.zeros((6,2))
    B[4,0] = 1/self.m
    B[4,1] = 1/self.m
    B[5,0] = self.a/self.I
    B[5,1] = -self.a/self.I
    """

    # Linearize about the target
    """
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, -cos(self.x_d[2]) * (self.u_d[0] + self.u_d[1]) / self.m, 0, 0, 0],
                  [0, 0, -sin(self.x_d[2]) * (self.u_d[0] + self.u_d[1]) / self.m, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [-sin(self.x_d[2]) / self.m, -sin(self.x_d[2]) / self.m],
                  [cos(self.x_d[2]) / self.m, cos(self.x_d[2]) / self.m],
                  [self.a / self.I, -self.a / self.I]])
    """

    # Lienarize about the current x and u
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, -cos(x_current[2]) * (u_current[0] + u_current[1]) / self.m, 0, 0, 0],
                  [0, 0, -sin(x_current[2]) * (u_current[0] + u_current[1]) / self.m, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [-sin(x_current[2]) / self.m, -sin(x_current[2]) / self.m],
                  [ cos(x_current[2]) / self.m,  cos(x_current[2]) / self.m],
                  [self.a / self.I, -self.a / self.I]])
    return A, B

  def discrete_time_linearized_dynamics(self, x_current, u_current):
    # Discrete time version of the linearized dynamics at the fixed point

    A_c, B_c = self.continuous_time_linearized_dynamics(x_current, u_current)

    # Manually discretize about origin
    """
    A_d = np.identity(6) + A_c * T
    B_d = B_c * T
    """

    # Discretize using scipy library
    C = np.eye(A_c.shape[0])
    D = np.zeros((A_c.shape[0],))
    [A_d, B_d, _, _, _] = cont2discrete((A_c, B_c, C, D), self.dt)

    return A_d, B_d

  def add_initial_state_constraint(self, prog, x, x_current):
    # impose initial state constraint.

    lb = x_current - self.x_d
    ub = x_current - self.x_d
    # lb = np.zeros((self.n_x))
    # ub = np.zeros((self.n_x))

    prog.AddBoundingBoxConstraint(lb, ub, x[0, :])

    pass

  def add_input_saturation_constraint(self, prog, x, u, N):
    # impose input limit constraint.

    lb = self.umin * np.ones(self.n_u) - self.u_d
    ub = self.umax * np.ones(self.n_u) - self.u_d
    for i in range(N-1):
        prog.AddBoundingBoxConstraint(lb, ub, u[i,])

    pass

  def add_dynamics_constraint(self, prog, x, u, N, x_current, u_current):
    # impose dynamics constraint.

    A, B = self.discrete_time_linearized_dynamics(x_current, u_current)
    lhs_eq = np.zeros((self.n_x, ))
    # lhs_eq = A @ self.x_d + B @ self.u_d - self.x_d
    # lhs_eq = A @ x_current + B @ u_current - x_current

    for i in range(N-1):
        prog.AddLinearEqualityConstraint(A @ x[i, :] + B @ u[i, :] - x[i+1, :],  np.zeros((self.n_x, )))
    pass

  def add_cost(self, prog, x, u, N):
    # add quadratic cost; no linear cost.

    b_u = np.zeros((self.n_u, 1))
    b_x = np.zeros((self.n_x, 1))
    # b_x =  - self.Q @ self.x_d

    for i in range(N-1):
        prog.AddQuadraticCost(self.Q, b_x, x[i])
        prog.AddQuadraticCost(self.R, b_u, u[i])
    prog.AddQuadraticCost(self.Qf, b_x, x[-1])
    pass

  def compute_mpc_feedback(self, x_current, u_current):
    # computes the MPC controller input u

    N = 20 # Time horizon

    # Initialize mathematical program and decalre decision variables
    prog = MathematicalProgram()
    x = np.zeros((N, 6), dtype="object")
    for i in range(N):
      x[i] = prog.NewContinuousVariables(6, "x_" + str(i))
    u = np.zeros((N-1, 2), dtype="object")
    for i in range(N-1):
      u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

    # Add constraints and cost
    self.add_initial_state_constraint(prog, x, x_current)
    self.add_input_saturation_constraint(prog, x, u, N)
    self.add_dynamics_constraint(prog, x, u, N, x_current, u_current)
    self.add_cost(prog, x, u, N)

    # Solve the QP
    solver = OsqpSolver()
    result = solver.Solve(prog)

    # retrieve the controller input from the solution of the optimization problem
    # and use it to compute the MPC input u
    
    u_series = result.GetSolution(u)
    u_mpc = u_series[0] + self.u_d

    x_series = result.GetSolution(x)
    x_mpc = x_series + self.x_d

    return u_mpc, x_mpc


  # def compute_lqr_feedback(self, x):
  #   '''
  #   Infinite horizon LQR controller
  #   '''
  #   A, B = self.continuous_time_linearized_dynamics()
  #   S = solve_continuous_are(A, B, self.Q, self.R)
  #   K = -inv(self.R) @ B.T @ S
  #   u = self.u_d() + K @ x
  #   return u
